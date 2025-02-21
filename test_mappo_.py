import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from scheduler_ import SurgeryQuotaScheduler
from tqdm import tqdm


# Define the Actor network (same as in training)
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.network(obs)


# MAPPO Agent for testing
class MAPPOAgent:
    def __init__(self, obs_dim, action_dim):
        self.actor = Actor(obs_dim, action_dim)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

    def get_action(self, obs):
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            probs = self.actor(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()


# MAPPO Tester
class MAPPOTester:

    def __init__(self, env, n_agents, obs_dim, action_dim, options=None):
        self.env = env
        self.options = options
        self.n_agents = n_agents
        self.agents = [MAPPOAgent(obs_dim, action_dim) for _ in range(n_agents)]

    def calculate_deviation(self, observed_states, target_state):
        num_episodes = len(observed_states)
        bootstrap_average_distribution = {i: sum(episode[i] for episode in observed_states) / num_episodes for i in
                                          range(7)}

        bootstrap_average_percentage_deviations = {i: np.nan for i in range(7)}  # Use np.nan as a default value
        mean_target_state = {day: (target_state[day]['max'] + target_state[day]['min']) / 2 for day in target_state}

        for key, value in bootstrap_average_distribution.items():
            target_value = mean_target_state[key]
            if target_value != 0:  # Avoid division by zero
                bootstrap_average_percentage_deviations[key] = np.abs(target_value - value) / target_value
            else:
                bootstrap_average_percentage_deviations[key] = np.nan  # Handle the zero case

        average_bootstrap_deviation = np.nanmean(
            list(bootstrap_average_percentage_deviations.values()))  # Use np.nanmean to ignore NaNs
        std_bootstrap_deviation = np.nanstd(
            list(bootstrap_average_percentage_deviations.values()))  # Use np.nanstd to ignore NaNs

        return average_bootstrap_deviation, std_bootstrap_deviation

    def test(self, n_episodes, max_steps, target_state):
        all_observed_states = []
        all_final_positions = []
        all_scaling_factors = []

        for e in tqdm(range(n_episodes), desc="Bootstrap testing..."):
            obs, info = self.env.reset(options=self.options)

            print(f'\nEpisode {e}:')
            print(self.env.render())
            print([(pos, sf) for pos, sf in zip([agent_info['position'] for agent_info in info.values()],
                                                [agent_info['scaling_factor'] for agent_info in
                                                 info.values()])])
            episode_observed_states = []

            for step in range(max_steps):
                actions = {}
                for i, agent in enumerate(self.agents):
                    action = agent.get_action(obs[f"agent_{i}"])
                    actions[f"agent_{i}"] = action

                next_obs, _, dones, truncations, info = self.env.step(actions)
                print(f'Step {step}', self.env.render())
                print([(pos, sf) for pos, sf in zip([agent_info['position'] for agent_info in info.values()],
                                                    [agent_info['scaling_factor'] for agent_info in info.values()])])

                if any(dones.values()) or any(truncations.values()):
                    episode_observed_states.append(self.env.observed_state)
                    all_final_positions.extend([agent_info['position'] for agent_info in info.values()])
                    all_scaling_factors.extend([agent_info['scaling_factor'] for agent_info in info.values()])

                obs = next_obs

                if all(dones.values()) or all(truncations.values()):
                    break

            all_observed_states.extend(episode_observed_states)

        mean_deviation, std_deviation = self.calculate_deviation(all_observed_states, target_state)
        avg_bids_per_day = {day: sum(state[day] for state in all_observed_states) / len(all_observed_states) for day in
                            range(7)}

        scaling_factor_positions = {}
        for sf, pos in zip(all_scaling_factors, all_final_positions):
            if sf not in scaling_factor_positions:
                scaling_factor_positions[sf] = []
            scaling_factor_positions[sf].append(pos)
        avg_position_per_scaling_factor = {sf: np.mean(positions) for sf, positions in scaling_factor_positions.items()}

        positions_sf_4 = scaling_factor_positions.get(4, [])
        positions_sf_6 = scaling_factor_positions.get(6, [])

        combined_positions = positions_sf_4 + positions_sf_6

        std_combined_positions = np.std(combined_positions)

        return mean_deviation, std_deviation, avg_bids_per_day, avg_position_per_scaling_factor, std_combined_positions

    def bootstrap_test(self, n_episodes, max_steps, target_state):
        mean_deviation, std_deviation, avg_bids, avg_positions, std_combined_positions = self.test(n_episodes, max_steps, target_state)

        print(f"Average number of bids per day: {avg_bids}")
        print(f"Average position per scaling factor: {avg_positions}")
        print(f"Operator preferences: {target_state}")
        print(f"Percentage deviation: mean = {mean_deviation:.2%}, std = {std_deviation: .2%}")
        print(f"4 & 6 SF std: {std_combined_positions}")

    def load_model(self, path):
        for i, agent in enumerate(self.agents):
            agent.load_model(os.path.join(path, f'actor_{i}.pth'))
        print(f"Loaded model from {path}")


# Main testing loop
if __name__ == "__main__":
    n_agents = 21
    env = SurgeryQuotaScheduler(render_mode='terminal', max_agents=n_agents)

    obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").n

    tester = MAPPOTester(env, n_agents, obs_dim, action_dim)
    tester.load_model(path='trained_model')
    tester.bootstrap_test(n_episodes=5, max_steps=7,
                          target_state={0: {'min': 5, 'max': 5},
                                        1: {'min': 5, 'max': 5},
                                        2: {'min': 4, 'max': 4},
                                        3: {'min': 4, 'max': 4},
                                        4: {'min': 3, 'max': 3},
                                        5: {'min': 0, 'max': 0},
                                        6: {'min': 0, 'max': 0}})
