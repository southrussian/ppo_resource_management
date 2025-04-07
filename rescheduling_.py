import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from scheduler_ import ResourceScheduler
import itertools
import logging
import matplotlib.pyplot as plt


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
        logging.info(f"Agent action: {action.item()}")
        return action.item()


class Client:
    def __init__(self, name, urgency, completeness, complexity) -> None:
        self.name = name
        self.urgency = urgency
        self.completeness = completeness
        self.complexity = complexity
        self.acceptance_rate = np.random.randint(25, 76) / 100
        self._satisfied = False
        self._assigned_agent = None

    @property
    def satisfied(self):
        return self._satisfied

    @satisfied.setter
    def satisfied(self, value):
        self._satisfied = value

    @property
    def assigned_agent(self):
        return self._assigned_agent

    @assigned_agent.setter
    def assigned_agent(self, value):
        self._assigned_agent = value

    def give_feedback(self):
        answer = np.random.choice([True, False], p=[self.acceptance_rate, 1 - self.acceptance_rate])
        if answer:
            self.acceptance_rate = 1.0
        return answer


class MultiAgentSystemOperator:
    def __init__(self, list_of_clients) -> None:
        self.clients = list_of_clients

    def collect_feedback(self):
        for client in self.clients:
            client.satisfied = client.give_feedback()

    def assign_agents(self, agents):
        for client in self.clients:
            health_state = (client.urgency, client.completeness, client.complexity)
            client.assigned_agent = agents[health_state]

    def get_actions(self, observations):
        actions = {}
        i = 0
        for client in self.clients:
            model = client.assigned_agent['model_file']
            logging.info(f"Observations for {client.name}: {observations[f'agent_{client.name[-1]}']}")
            actions[f'agent_{i}'] = model.get_action(observations[f'agent_{client.name[-1]}'])
            i += 1
        return actions


# def plot_histogram(data, episode_num):
#     days = list(data.keys())
#     values = list(data.values())
#
#     plt.figure(figsize=(10, 6))
#     plt.bar(days, values, color='skyblue')
#     plt.xlabel('Week Days')
#     plt.ylabel('Values of occupied slots')
#     plt.title(f'Histogram of slots distribution of episode: {episode_num+1}')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig(f'histogram_episode_{episode_num+1}.png')
#     plt.show()


def calculate_deviation(observed_states, target_state):
    num_episodes = len(observed_states)
    bootstrap_average_distribution = {i: sum(episode[i] for episode in observed_states) / num_episodes for i in
                                      range(7)}

    bootstrap_average_percentage_deviations = {i: np.nan for i in range(7)}  # Use np.nan as a default value
    mean_target_state = {day: (target_state[day]['min'] + target_state[day]['max']) / 2 for day in target_state}

    for key, value in bootstrap_average_distribution.items():
        target_value = mean_target_state[key]
        if target_value != 0:  # Avoid division by zero
            bootstrap_average_percentage_deviations[key] = np.abs(target_value - value) / target_value
        else:
            bootstrap_average_percentage_deviations[key] = np.nan

    average_bootstrap_deviation = np.nanmean(
        list(bootstrap_average_percentage_deviations.values()))
    std_bootstrap_deviation = np.nanstd(
        list(bootstrap_average_percentage_deviations.values()))

    return average_bootstrap_deviation, std_bootstrap_deviation


def calculate_scaling_factor_positions(observed_states):
    scaling_factor_positions = {i: [] for i in range(7)}

    for state in observed_states:
        for day, value in state.items():
            scaling_factor_positions[day].append(value)

    average_position_per_scaling_factor = {day: np.mean(values) for day, values in scaling_factor_positions.items()}
    return average_position_per_scaling_factor


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("logfile.log"),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()

    # Input for number of patients
    while True:
        try:
            num_patients = int(input("Enter the number of patients: "))
            if num_patients > 0:
                break
            else:
                print("The number of patients must be a positive integer. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid integer for the number of patients.")

    # Input for patient attributes
    patients = []
    for i in range(num_patients):
        while True:
            try:
                urgency = int(input(f"Enter urgency for patient {i+1} (1-3): "))
                if 1 <= urgency <= 3:
                    break
                else:
                    print("Urgency must be an integer between 1 and 3. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid integer for urgency.")

        while True:
            try:
                completeness = int(input(f"Enter completeness for patient {i+1} (0-1): "))
                if 0 <= completeness <= 1:
                    break
                else:
                    print("Completeness must be an integer between 0 and 1. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid integer for completeness.")

        while True:
            try:
                complexity = int(input(f"Enter complexity for patient {i+1} (0-1): "))
                if 0 <= complexity <= 1:
                    break
                else:
                    print("Complexity must be an integer between 0 and 1. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid integer for complexity.")

        patients.append(Client(name=f'client_{i}', urgency=urgency, completeness=completeness, complexity=complexity))

    urgency_range = range(1, 4)
    completeness_range = range(0, 2)
    complexity_range = range(0, 2)

    options = list(itertools.product(urgency_range, completeness_range, complexity_range))

    selected_states = np.random.choice(len(options), size=12, replace=False)
    health_states = [options[i] for i in selected_states]

    agents = {f'agent_{i}': MAPPOAgent(obs_dim=11, action_dim=3) for i in range(len(health_states))}

    for i, agent in enumerate(agents):
        agents[agent].load_model(f'trained_model/actor_{i}.pth')

    manager = MultiAgentSystemOperator(list_of_clients=patients)
    manager.assign_agents(
        {health_state: {'agent_name': agent_name, 'model_file': model_file} for health_state, (agent_name, model_file)
         in zip(health_states, agents.items())})

    e = 0
    all_observed_states = []
    last_episode_agent_positions = None

    target_state = {}
    for day in range(7):
        while True:
            try:
                min_val = int(input(f"Enter the minimum target value for day {day}: "))
                max_val = int(input(f"Enter the maximum target value for day {day}: "))
                if min_val <= max_val:
                    target_state[day] = {'min': min_val, 'max': max_val}
                    break
                else:
                    print("Minimum value must be less than or equal to the maximum value. Please try again.")
            except ValueError:
                print("Invalid input. Please enter valid integers for the minimum and maximum target values.")

    # Input for initial positions
    initial_positions = {}
    for i in range(num_patients):
        while True:
            try:
                position = int(input(f"Enter initial position for patient {i+1} (0-6): "))
                if 0 <= position <= 6:
                    initial_positions[f'agent_{i}'] = position
                    break
                else:
                    print("Position must be an integer between 0 and 6. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid integer for the position.")

    while not all([client.satisfied for client in manager.clients]):
        logger.info(f"Starting episode {e}")

        env = ResourceScheduler(render_mode='terminal', max_agents=len(patients),
                                max_days=7, max_episode_length=7)
        obs, _ = env.reset(
            options={
                'target_state': target_state,
                'agents_data': {f'agent_{i}': {'active': ~client.satisfied,
                                               'base_reward': 1.0,
                                               'window': 3,
                                               'alpha': 2.0,
                                               'urgency': client.urgency,
                                               'completeness': client.completeness,
                                               'complexity': client.complexity,
                                               'mutation_rate': 0.0}
                                for i, client in enumerate(manager.clients)},
                'initial_positions': initial_positions
            }
        )

        logger.info(f"Episode {e} - {env.render()}")

        episode_observed_states = []

        while True:
            actions = manager.get_actions(observations=obs)
            logger.debug(f"Actions: {actions}")
            obs, _, dones, truncations, _ = env.step(actions)

            logger.info(f"Episode {e} - {env.render()}")

            if any(dones.values()) or any(truncations.values()):
                rendered_data = env.render()
                episode_observed_states.append(env.observed_state)

                # Собираем данные о позициях агентов для текущего эпизода
                current_episode_agent_positions = {}
                for agent_name in env.agents:
                    position = env.agents_data[agent_name]['position']
                    scaling_factor = env.agents_data[agent_name]['scaling_factor']
                    if position not in current_episode_agent_positions:
                        current_episode_agent_positions[position] = []
                    current_episode_agent_positions[position].append(f"{agent_name} scaling factor: {scaling_factor}")
                last_episode_agent_positions = current_episode_agent_positions  # Сохраняем последний эпизод

                break

        env.close()

        all_observed_states.extend(episode_observed_states)
        manager.collect_feedback()

        e += 1

    mean_deviation, std_deviation = calculate_deviation(all_observed_states, target_state)
    avg_bids_per_day = {i: sum(state[i] for state in all_observed_states) / len(all_observed_states) for i in range(7)}

    avg_position_per_scaling_factor = calculate_scaling_factor_positions(all_observed_states)

    operator_preferences = {day: {'min': target_state[day]['min'], 'max': target_state[day]['max']} for day in range(7)}

    print(f"Average number of bids per day: {avg_bids_per_day}")
    print(f"Average position per scaling factor: {avg_position_per_scaling_factor}")
    print(f"Operator preferences: {operator_preferences}")
    print(f"Percentage deviation: mean = {mean_deviation * 100:.2f}%, std =  {std_deviation * 100:.2f}%")

    print("\nAgent distributions by days in the last episode:")
    for day in sorted(last_episode_agent_positions.keys()):
        agents = last_episode_agent_positions[day]
        print(f"Day {day}: {', '.join(agents)}")
