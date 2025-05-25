import functools

import gymnasium
import numpy as np
import pygame
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv
from scipy.stats import levy_stable


class ResourceScheduler(ParallelEnv):
    metadata = {'render_modes': ['human', 'terminal'],
                'name': 'sqsc_v1'}

    def __init__(self, render_mode=None, max_agents=12, max_days=7, max_episode_length=7):
        self.max_agents = max_agents
        self.max_days = max_days
        self.max_episode_length = max_episode_length
        self.possible_agents = ["agent_" + str(r) for r in range(self.max_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.agent_action_mapping = {0: 1, 1: -1, 2: 0}
        self.agent_action_mapping_text = {0: 'переместиться на следующий слот', 1: 'переместиться на предыдущий слот',
                                          2: 'остаться на текущем слоте'}
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([3, 2, 2, self.max_days, *[self.max_agents + 2 for _ in range(self.max_days)]],
                             start=[1, 0, 0, 0, *[-1 for _ in range(self.max_days)]])

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def render(self):
        if self.render_mode == "human":
            if not hasattr(self, 'pygame_initialized'):
                pygame.init()
                self.win_width, self.win_height = 1000, 500
                self.win = pygame.display.set_mode((self.win_width, self.win_height))
                pygame.display.set_caption("Симуляция распределения ресурсов")
                self.pygame_initialized = True

            # Colors
            BACKGROUND = (240, 240, 240)
            DAY_BG = (220, 220, 255)
            AGENT_BG = (200, 255, 200)
            TEXT_COLOR = (0, 0, 0)
            LINE_COLOR = (100, 100, 100)
            TITLE_COLOR = (50, 50, 150)

            # Fonts
            title_font = pygame.font.Font(None, 48)
            day_font = pygame.font.Font(None, 36)
            agent_font = pygame.font.Font(None, 32)

            self.win.fill(BACKGROUND)

            # Draw title
            title = title_font.render("Симуляция распределения ресурсов", True, TITLE_COLOR)
            self.win.blit(title, (self.win_width // 2 - title.get_width() // 2, 20))

            # Calculate positions for days with automatic spacing
            max_days = self.max_days
            day_width = 100
            day_height = 100
            total_days_width = max_days * day_width
            available_width = self.win_width - 100  # 50px padding on each side
            spacing = (available_width - total_days_width) // (max_days + 1) if max_days > 1 else 0
            start_x = (self.win_width - (total_days_width + (max_days - 1) * spacing)) // 2
            start_y = 100

            # Draw days
            for day in range(max_days):
                x = start_x + day * (day_width + spacing)
                y = start_y

                # Draw day rectangle
                pygame.draw.rect(self.win, DAY_BG, (x, y, day_width, day_height))
                pygame.draw.rect(self.win, LINE_COLOR, (x, y, day_width, day_height), 2)

                # Draw day label
                day_label = day_font.render(f"День {day + 1}", True, TEXT_COLOR)
                self.win.blit(day_label, (x + day_width // 2 - day_label.get_width() // 2, y + 10))

                # Draw agents in this day
                agents_in_day = [agent for agent in self.agents
                                 if self.agents_data[agent]['position'] == day]

                if agents_in_day:
                    agent_y = y + 50
                    max_agents_per_column = 3
                    agent_height = 25

                    for i, agent in enumerate(agents_in_day):
                        if i >= max_agents_per_column:
                            more_text = agent_font.render(f"+{len(agents_in_day) - max_agents_per_column}", True,
                                                          TEXT_COLOR)
                            self.win.blit(more_text, (x + day_width // 2 - more_text.get_width() // 2, agent_y))
                            break

                        agent_rect = pygame.Rect(x + 10, agent_y, day_width - 20, agent_height)
                        pygame.draw.rect(self.win, AGENT_BG, agent_rect)
                        pygame.draw.rect(self.win, LINE_COLOR, agent_rect, 1)

                        agent_text = agent_font.render(agent.split('_')[1], True, TEXT_COLOR)
                        self.win.blit(agent_text,
                                      (x + day_width // 2 - agent_text.get_width() // 2,
                                       agent_y + 5))
                        agent_y += 30

            # Draw legend
            legend_y = start_y + day_height + 50
            legend_title = day_font.render("Состояние занятости слотов (Текущее/Целевое):", True, TEXT_COLOR)
            self.win.blit(legend_title, (self.win_width // 2 - legend_title.get_width() // 2, legend_y))

            # Draw target vs actual
            target_y = legend_y + 40
            for day in range(max_days):
                x = start_x + day * (day_width + spacing)
                target = self.target_state.get(day, 0)
                actual = self.observed_state.get(day, 0)

                target_text = agent_font.render(f"{actual}/{target}", True,
                                                TEXT_COLOR if actual == target
                                                else (255, 0, 0) if actual > target
                                                else (0, 150, 0))
                self.win.blit(target_text, (x + day_width // 2 - target_text.get_width() // 2, target_y))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.pygame_initialized = False
                    return

        elif self.render_mode == 'terminal':
            return self.observed_state
        else:
            gymnasium.logger.warn('You are calling render mode without specifying any render mode.')
            return

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.target_state = {day: (boundaries['min'] + boundaries['max']) / 2 for day, boundaries in
                             options['target_state'].items()} if options is not None else {0: 4, 1: 2, 2: 3, 3: 3, 4: 0,
                                                                                           5: 0, 6: 0}
        self.agents_data = options['agents_data'] if options is not None else {agent: {'active': True,
                                                                                       'base_reward': 1.0,
                                                                                       'window': 3,
                                                                                       'alpha': 2.0,
                                                                                       'urgency': self.rng.integers(1,
                                                                                                                    4,
                                                                                                                    1).item(),
                                                                                       'completeness': self.rng.integers(
                                                                                           0, 2, 1).item(),
                                                                                       'complexity': self.rng.integers(
                                                                                           0, 2, 1).item(),
                                                                                       'mutation_rate': 0.0} for agent
                                                                               in self.agents}
        for agent in self.agents:
            self.agents_data[agent]['position'] = self.rng.integers(0, 7, 1).item()
        for agent in self.agents:
            self.agents_data[agent]['scaling_factor'] = max(1, (self.agents_data[agent]['complexity'] +
                                                                (1 - self.agents_data[agent]['completeness'])) *
                                                            self.agents_data[agent]['urgency'])
        self.num_moves = 0
        observations = {agent: np.array([self.agents_data[agent]['urgency'],
                                         self.agents_data[agent]['completeness'],
                                         self.agents_data[agent]['complexity'],
                                         self.agents_data[agent]['position'],
                                         *[self.observed_state.get(self.agents_data[agent]['position'] +
                                                                   np.ceil(d - self.agents_data[agent]['window'] / 2),
                                                                   -1)
                                           for d in range(self.max_days)]])
                        for agent in self.agents}
        infos = {agent: self.agents_data[agent] for agent in self.agents}
        return observations, infos

    def step(self, actions):
        if not actions:
            return {}, {}, {}, {}, {}
        self.num_moves += 1
        for agent in self.agents:
            if self.agents_data[agent]['active']:
                self.agents_data[agent]['position'] = (self.agents_data[agent]['position'] + self.agent_action_mapping[
                    int(actions[agent])]) % self.max_days
                if self.agents_data[agent]['position'] > (self.max_days // 2):
                    if int(actions[agent]) == 0:
                        if self.agents_data[agent]['mutation_rate'] != 1.0:
                            self.agents_data[agent]['mutation_rate'] = min(
                                self.agents_data[agent]['mutation_rate'] + 0.05, 1.0)
                    elif int(actions[agent]) == 1:
                        if self.agents_data[agent]['mutation_rate'] != 0.0:
                            self.agents_data[agent]['mutation_rate'] = max(
                                self.agents_data[agent]['mutation_rate'] - 0.05, 0.0)
        rewards = {agent: self.reward_map(agent) for agent in self.agents}
        terminations = {agent: list(actions.values()).count(2) > self.max_agents * 0.8 for agent in self.agents}
        truncations = {agent: self.num_moves >= self.max_episode_length for agent in self.agents}
        if any(terminations.values()) or any(truncations.values()):
            rewards = {agent: r - 0.5 * abs(
                self.observed_state[self.agents_data[agent]['position']] - self.target_state[
                    self.agents_data[agent]['position']]) for agent, r in rewards.items()}
        observations = {agent: np.array([self.agents_data[agent]['urgency'],
                                         self.agents_data[agent]['completeness'],
                                         self.agents_data[agent]['complexity'],
                                         self.agents_data[agent]['position'],
                                         *[self.observed_state.get(self.agents_data[agent]['position'] +
                                                                   np.ceil(d - self.agents_data[agent]['window'] / 2),
                                                                   -1)
                                           for d in range(self.max_days)]]) for agent in self.agents}
        infos = {agent: self.agents_data[agent] for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    @property
    def observed_state(self):
        return {day: [self.agents_data[agent]['position'] for agent in self.agents].count(day) for day in
                range(self.max_days)}

    def reward_map(self, agent):
        discrepancy = {day: abs(self.observed_state[day] - self.target_state[day]) for day in self.target_state}
        window = [int((self.agents_data[agent]['position'] + np.ceil(
            d - self.agents_data[agent]['window'] / 2)) % self.max_days) for d in
                  range(self.agents_data[agent]['window'])]
        masked_discrepancy = {day: discrepancy[day] if day in window else 0 for day in discrepancy}
        rv = levy_stable(self.agents_data[agent]['alpha'], 0.0, loc=self.agents_data[agent]['position'], scale=1.0)
        weighted_discrepancy_sum = np.sum([rv.pdf(day) * masked_discrepancy[day] for day in masked_discrepancy])

        global_discrepancy = sum(discrepancy.values())

        reward = -(weighted_discrepancy_sum + 0.05 * global_discrepancy) * self.agents_data[agent]['base_reward'] / \
                 self.agents_data[agent]['scaling_factor']

        current_day = self.agents_data[agent]['position']
        if self.observed_state[current_day] < self.target_state[current_day]:
            reward += 0.5 * self.agents_data[agent]['base_reward']  # Increased reward

        if self.observed_state[current_day] > self.target_state[current_day]:
            reward -= 0.1 * self.agents_data[agent]['base_reward']  # Reduced penalty

        return reward
