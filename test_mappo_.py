import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from scheduler_ import ResourceScheduler
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
        self.logs = []

    @staticmethod
    def calculate_deviation(observed_states, target_state):
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

    def _log_agent_step(self, agent_id, episode, step, obs, action, info, next_info):
        current_position = info['position']
        prev_day = (current_position - 1) % 7
        next_day = (current_position + 1) % 7

        return {
            "agent_id": agent_id,
            "episode": episode,
            "step": step,
            "beliefs": {
                "urgency": info['urgency'],
                "completeness": info['completeness'],
                "complexity": info['complexity'],
                "current_position": current_position,
                "slot_occupancy_prev": self.env.observed_state.get(prev_day, 0),
                "slot_occupancy_current": self.env.observed_state.get(current_position, 0),
                "slot_occupancy_next": self.env.observed_state.get(next_day, 0)
            },
            "desires": [
                "optimize_surgery_day",
                "reduce_conflicts",
                "balance_workload"
            ],
            "intention": action,
            "state_after_action": {
                "new_position": next_info['position'],
                "scaling_factor": next_info['scaling_factor'],
                "current_day": next_info['position']
            }
        }

    def test(self, n_episodes, max_steps, target_state):
        all_observed_states = []
        all_final_positions = []
        all_scaling_factors = []

        self.logs = []
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
                step_logs = []
                for i, agent in enumerate(self.agents):
                    agent_obs = obs[f"agent_{i}"]
                    action = agent.get_action(obs[f"agent_{i}"])
                    actions[f"agent_{i}"] = action
                    step_logs.append({
                        'agent_id': i,
                        'pre_info': info[f"agent_{i}"]
                    })

                next_obs, _, dones, truncations, next_info = self.env.step(actions)

                # Log post-step state
                for log in step_logs:
                    agent_id = log['agent_id']
                    log_entry = self._log_agent_step(
                        agent_id=agent_id,
                        episode=e,
                        step=step,
                        obs=obs[f"agent_{agent_id}"],
                        action=actions[f"agent_{agent_id}"],
                        info=log['pre_info'],
                        next_info=next_info[f"agent_{agent_id}"]
                    )
                    self.logs.append(log_entry)

                print(f'Step {step}', self.env.render())
                print([(pos, sf) for pos, sf in zip([agent_info['position'] for agent_info in info.values()],
                                                    [agent_info['scaling_factor'] for agent_info in info.values()])])

                if any(dones.values()) or any(truncations.values()):
                    episode_observed_states.append(self.env.observed_state)
                    all_final_positions.extend([agent_info['position'] for agent_info in info.values()])
                    all_scaling_factors.extend([agent_info['scaling_factor'] for agent_info in info.values()])

                obs, info = next_obs, next_info

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

    def format_logs_for_llm(self):
        formatted = []
        for log in self.logs:
            formatted.append(
                f"Agent {log['agent_id']} [Episode {log['episode']} Step {log['step']}]:\n"
                f"Beliefs:\n"
                f"- Urgency: {log['beliefs']['urgency']} | "
                f"Completeness: {log['beliefs']['completeness']} | "
                f"Complexity: {log['beliefs']['complexity']}\n"
                f"- Position: {log['beliefs']['current_position']} | "
                f"Slots: Prev({log['beliefs']['slot_occupancy_prev']}) "
                f"Current({log['beliefs']['slot_occupancy_current']}) "
                f"Next({log['beliefs']['slot_occupancy_next']})\n"
                f"Intention: Action {log['intention']} ({self.env.agent_action_mapping[log['intention']]})\n"
                f"Result: New Position {log['state_after_action']['new_position']} | "
                f"SF: {log['state_after_action']['scaling_factor']} | "
                f"Day: {log['state_after_action']['current_day']}\n"
            )
        return "\n".join(formatted)


# Main testing loop
if __name__ == "__main__":
    n_agents = 12
    env = ResourceScheduler(render_mode='terminal', max_agents=n_agents)

    obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").n

    tester = MAPPOTester(env, n_agents, obs_dim, action_dim)
    tester.load_model(path='trained_model')
    tester.bootstrap_test(n_episodes=5, max_steps=7,
                          target_state={0: {'min': 4, 'max': 4},
                                        1: {'min': 4, 'max': 4},
                                        2: {'min': 2, 'max': 2},
                                        3: {'min': 2, 'max': 2},
                                        4: {'min': 0, 'max': 0},
                                        5: {'min': 0, 'max': 0},
                                        6: {'min': 0, 'max': 0}})

    llm_logs = tester.format_logs_for_llm()
    print("\nSystem logs for LLM analysis:")
    print(llm_logs)
