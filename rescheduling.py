import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from scheduler import ResourceScheduler
import itertools
import logging
import matplotlib.pyplot as plt

from service.yandex_explainability import yandex_explain
from service.gigachat_explainability import gigachat_explain


class Actor(nn.Module):
    """
    A neural network model representing the Actor in a reinforcement learning setup.

    This class defines the architecture of the Actor model, which is used to determine
    the action probabilities given the observation inputs.

    Attributes:
        network (nn.Sequential): The neural network architecture consisting of linear layers and ReLU activations.
    """
    def __init__(self, obs_dim, action_dim):
        """
        Initializes the Actor model with specified observation and action dimensions.

        Args:
            obs_dim (int): The dimension of the observation space.
            action_dim (int): The dimension of the action space.
        """
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
        """
        Performs a forward pass through the network to compute action probabilities.

        Args:
            obs (torch.Tensor): The input observation tensor.

        Returns:
            torch.Tensor: The output tensor representing action probabilities.
        """
        return self.network(obs)


class MAPPOAgent:
    """
    An agent that uses the Multi-Agent Proximal Policy Optimization (MAPPO) algorithm.

    This class manages the Actor model, loading its state, and selecting actions based on observations.

    Attributes:
        actor (Actor): The Actor model used to select actions.
    """
    def __init__(self, obs_dim, action_dim):
        """
        Initializes the MAPPOAgent with specified observation and action dimensions.

        Args:
            obs_dim (int): The dimension of the observation space.
            action_dim (int): The dimension of the action space.
        """
        self.actor = Actor(obs_dim, action_dim)

    def load_model(self, path):
        """
        Loads the Actor model's state from a file.

        Args:
            path (str): The file path to load the model state from.
        """
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

    def get_action(self, obs):
        """
        Selects an action based on the given observation using the Actor model.

        Args:
            obs (np.ndarray): The observation array.

        Returns:
            int: The selected action as an integer.
        """
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            probs = self.actor(obs)
        dist = Categorical(probs)
        action = dist.sample()
        logging.info(f"Agent action: {action.item()}")
        return action.item()


class Client:
    """
    Represents a client in the scheduling system.

    This class holds information about the client's attributes and manages their satisfaction state.

    Attributes:
        name (str): The name of the client.
        urgency (int): The urgency level of the client's request.
        completeness (int): The completeness level of the client's information.
        complexity (int): The complexity level of the client's task.
        acceptance_rate (float): The client's acceptance rate.
        _satisfied (bool): Whether the client is satisfied.
        _assigned_agent (MAPPOAgent): The agent assigned to the client.
    """
    def __init__(self, name, urgency, completeness, complexity) -> None:
        """
        Initializes the Client with specified attributes.

        Args:
            name (str): The name of the client.
            urgency (int): The urgency level of the client's request.
            completeness (int): The completeness level of the client's information.
            complexity (int): The complexity level of the client's task.
        """
        self.name = name
        self.urgency = urgency
        self.completeness = completeness
        self.complexity = complexity
        self.acceptance_rate = np.random.randint(25, 76) / 100
        self._satisfied = False
        self._assigned_agent = None

    @property
    def satisfied(self):
        """
        Gets the satisfaction state of the client.

        Returns:
            bool: The satisfaction state.
        """
        return self._satisfied

    @satisfied.setter
    def satisfied(self, value):
        """
        Sets the satisfaction state of the client.

        Args:
            value (bool): The satisfaction state to set.
        """
        self._satisfied = value

    @property
    def assigned_agent(self):
        """
        Gets the agent assigned to the client.

        Returns:
            MAPPOAgent: The assigned agent.
        """
        return self._assigned_agent

    @assigned_agent.setter
    def assigned_agent(self, value):
        """
        Sets the agent assigned to the client.

        Args:
            value (MAPPOAgent): The agent to assign.
        """
        self._assigned_agent = value

    def give_feedback(self):
        """
        Simulates the client giving feedback.

        Returns:
            bool: A random boolean representing the feedback.
        """
        answer = np.random.choice([True, False], p=[self.acceptance_rate, 1 - self.acceptance_rate])
        if answer:
            self.acceptance_rate = 1.0
        return answer


class MultiAgentSystemOperator:
    """
    Manages a multi-agent system, handling client feedback, agent assignment, and logging.

    This class is responsible for collecting feedback from clients, assigning agents to clients based on their cargo state,
    and logging agent steps during the simulation.

    Attributes:
        clients (list): A list of Client objects representing the clients in the system.
        logs (list): A list to store logs of agent steps and actions.
    """
    def __init__(self, list_of_clients) -> None:
        """
        Initializes the MultiAgentSystemOperator with a list of clients.

        Args:
            list_of_clients (list): A list of Client objects.
        """
        self.clients = list_of_clients
        self.logs = []

    def collect_feedback(self):
        """
        Collects feedback from all clients and updates their satisfaction status.

        This method iterates over each client and sets their satisfaction status based on the feedback they provide.
        """
        for client in self.clients:
            client.satisfied = client.give_feedback()

    def assign_agents(self, agents):
        """
        Assigns agents to clients based on their cargo state.

        Args:
            agents (dict): A dictionary mapping cargo states to agents.

        This method iterates over each client and assigns an agent based on the client's urgency, completeness,
        and complexity attributes.
        """
        for client in self.clients:
            cargo_state = (client.urgency, client.completeness, client.complexity)
            client.assigned_agent = agents[cargo_state]

    def _log_agent_step(self, agent_id, episode, step, action, info, next_info, env):
        """
        Logs the step taken by an agent, including beliefs, desires, intentions, and state after action.

        Args:
            agent_id (int): The ID of the agent.
            episode (int): The current episode number.
            step (int): The current step number within the episode.
            action (int): The action taken by the agent.
            info (dict): Information about the agent's state before the action.
            next_info (dict): Information about the agent's state after the action.
            env: The environment in which the agent is operating.

        This method constructs a log entry with detailed information about the agent's step and appends it to the logs list.
        """
        current_position = info['position']
        prev_day = (current_position - 1) % 7
        next_day = (current_position + 1) % 7

        log_entry = {
            "agent_id": agent_id,
            "episode": episode,
            "step": step,
            "beliefs": {
                "urgency": info['urgency'],
                "completeness": info['completeness'],
                "complexity": info['complexity'],
                "current_position": current_position,
                "slot_occupancy_prev": env.observed_state.get(prev_day, 0),
                "slot_occupancy_current": env.observed_state.get(current_position, 0),
                "slot_occupancy_next": env.observed_state.get(next_day, 0)
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
        self.logs.append(log_entry)

    def get_actions(self, observations, env, episode, step):
        """
        Gets actions for all agents based on the current observations and logs the steps.

        Args:
            observations (dict): Observations for each agent.
            env: The environment in which the agents are operating.
            episode (int): The current episode number.
            step (int): The current step number within the episode.

        Returns:
            tuple: A tuple containing a dictionary of actions and a list of step logs.

        This method iterates over each client, retrieves the action from the assigned agent's model,
        and logs the step information.
        """
        actions = {}
        step_logs = []

        for i, client in enumerate(self.clients):
            agent_id = int(client.name.split('_')[-1])
            pre_info = env.agents_data[f'agent_{agent_id}']

            model = client.assigned_agent['model_file']
            action = model.get_action(observations[f'agent_{agent_id}'])
            actions[f'agent_{i}'] = action

            step_logs.append({
                'agent_id': agent_id,
                'pre_info': pre_info,
                'action': action
            })

        return actions, step_logs


def plot_histogram(data, episode_num):
    """
    Plots and saves a histogram of the distribution of occupied slots for a given episode.

    This function creates a bar chart representing the distribution of occupied slots over the days of the week
    for a specific episode and saves the plot as an image file.

    Args:
        data (dict): A dictionary where keys are days of the week and values are the number of occupied slots.
        episode_num (int): The episode number for which the histogram is being plotted.

    Returns:
        None
    """
    days = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(10, 6))
    plt.bar(days, values, color='skyblue')
    plt.xlabel('Week Days')
    plt.ylabel('Values of occupied slots')
    plt.title(f'Histogram of slots distribution of episode: {episode_num+1}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'histogram_episode_{episode_num+1}.png')
    plt.show()


def calculate_deviation(observed_states, target_state):
    """
    Calculates the average and standard deviation of percentage deviations between observed and target states.

    This function computes the mean and standard deviation of the percentage deviations of the observed states
    from the target state across all episodes.

    Args:
        observed_states (list): A list of dictionaries representing the observed states for each episode.
        target_state (dict): A dictionary representing the target state with 'min' and 'max' values for each day.

    Returns:
        tuple: A tuple containing the average bootstrap deviation and the standard deviation of the bootstrap deviations.
    """
    num_episodes = len(observed_states)
    bootstrap_average_distribution = \
        {i: sum(episode[i] for episode in observed_states) / num_episodes for i in range(7)}

    bootstrap_average_percentage_deviations = {k: np.nan for k in range(7)}
    mean_target_state = {day: (target_state[day]['min'] + target_state[day]['max']) / 2 for day in target_state}

    for key, value in bootstrap_average_distribution.items():
        target_value = mean_target_state[key]
        if target_value != 0:
            bootstrap_average_percentage_deviations[key] = np.abs(target_value - value) / target_value
        else:
            bootstrap_average_percentage_deviations[key] = np.nan

    average_bootstrap_deviation = np.nanmean(list(bootstrap_average_percentage_deviations.values()))
    std_bootstrap_deviation = np.nanstd(list(bootstrap_average_percentage_deviations.values()))

    return average_bootstrap_deviation, std_bootstrap_deviation

def calculate_scaling_factor_positions(observed_states):
    """
    Calculates the average position per scaling factor for each day of the week.

    This function computes the average position of agents for each day of the week based on the observed states
    across all episodes.

    Args:
        observed_states (list): A list of dictionaries representing the observed states for each episode.

    Returns:
        dict: A dictionary where keys are days of the week and values are the average positions per scaling factor.
    """
    scaling_factor_positions = {i: [] for i in range(7)}

    for state in observed_states:
        for day, value in state.items():
            scaling_factor_positions[day].append(value)

    average_position_per_scaling_factor = {day: np.mean(values) for day, values in scaling_factor_positions.items()}
    return average_position_per_scaling_factor

def format_logs_for_llm(logs, env, agent_id):
    """
    Formats logs for a specific agent to be used with a Large Language Model (LLM).

    This function processes the logs and formats them into a human-readable string that describes the agent's beliefs,
    intentions, and the results of its actions for each step in an episode. The formatted logs are intended to be used
    as input for an LLM to provide explanations or insights.

    Args:
        logs (list): A list of log entries, where each entry is a dictionary containing details about an agent's step.
        env: The environment in which the agents are operating, containing mappings for agent actions.
        agent_id (int): The ID of the agent whose logs are to be formatted.

    Returns:
        str: A formatted string containing the logs for the specified agent, ready to be used with an LLM.

    Notes:
        - The function filters logs based on the agent_id and formats only those logs that match the specified agent.
        - The formatted string includes details about the agent's beliefs, intentions, and the state after each action.
    """
    formatted = []
    for log in logs:
        if log['agent_id'] == agent_id:
            formatted.append(

                f"Агент {log['agent_id']} [Эпизод {log['episode']} Шаг {log['step']}]:\n"
                f"Убеждения:\n"
                f"- Срочность: {log['beliefs']['urgency']} | "
                f"Полнота информации: {log['beliefs']['completeness']} | "
                f"Сложность: {log['beliefs']['complexity']}\n"
                f"- Позиция: {log['beliefs']['current_position']} | "
                f"Слоты: Предыдущий({log['beliefs']['slot_occupancy_prev']}) "
                f"Текущий({log['beliefs']['slot_occupancy_current']}) "
                f"Следующий({log['beliefs']['slot_occupancy_next']})\n"
                f"Намерения: Действие {log['intention']} ({env.agent_action_mapping_text[log['intention']]})\n"
                f"Результат: Новая позиция {log['state_after_action']['new_position']} | "
                f"Коэфф. масштабирования: {log['state_after_action']['scaling_factor']} | "
                f"День: {log['state_after_action']['current_day']}\n"

            )
    return "\n".join(formatted)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("logfile.log"),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()

    while True:
        try:
            num_clients = int(input("Enter the number of clients: "))
            if num_clients > 0:
                break
            else:
                print("The number of clients must be a positive integer. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid integer for the number of clients.")

    clients = []
    for i in range(num_clients):
        while True:
            try:
                urgency = int(input(f"Enter urgency for client {i + 1} (1-3): "))
                if 1 <= urgency <= 3:
                    break
                else:
                    print("Urgency must be an integer between 1 and 3. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid integer for urgency.")

        while True:
            try:
                completeness = int(input(f"Enter completeness for client {i + 1} (0-1): "))
                if 0 <= completeness <= 1:
                    break
                else:
                    print("Completeness must be an integer between 0 and 1. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid integer for completeness.")

        while True:
            try:
                complexity = int(input(f"Enter complexity for client {i + 1} (0-1): "))
                if 0 <= complexity <= 1:
                    break
                else:
                    print("Complexity must be an integer between 0 and 1. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid integer for complexity.")

        clients.append(Client(name=f'client_{i}', urgency=urgency, completeness=completeness, complexity=complexity))

    urgency_range = range(1, 4)
    completeness_range = range(0, 2)
    complexity_range = range(0, 2)

    options = list(itertools.product(urgency_range, completeness_range, complexity_range))

    selected_states = np.random.choice(len(options), size=12, replace=False)
    cargo_states = [options[i] for i in selected_states]

    agents = {f'agent_{i}': MAPPOAgent(obs_dim=11, action_dim=3) for i in range(len(cargo_states))}

    for i, agent in enumerate(agents):
        agents[agent].load_model(f'trained_model/actor_{i}.pth')

    manager = MultiAgentSystemOperator(list_of_clients=clients)
    manager.assign_agents(
        {cargo_state: {'agent_name': agent_name, 'model_file': model_file} for cargo_state, (agent_name, model_file)
         in zip(cargo_states, agents.items())})

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

    initial_positions = {}
    for i in range(num_clients):
        while True:
            try:
                position = int(input(f"Enter initial position for client {i + 1} (0-6): "))
                if 0 <= position <= 6:
                    initial_positions[f'agent_{i}'] = position
                    break
                else:
                    print("Position must be an integer between 0 and 6. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid integer for the position.")

    while not all([client.satisfied for client in manager.clients]):
        logger.info(f"Starting episode {e}")

        env = ResourceScheduler(render_mode='terminal', max_agents=len(clients),
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

        episode_logs = []
        step = 0

        logger.info(f"Episode {e} - {env.render()}")

        episode_observed_states = []

        while True:
            actions, step_logs = manager.get_actions(observations=obs, env=env, episode=e, step=step)
            logger.debug(f"Actions: {actions}")
            next_obs, _, dones, truncations, next_info = env.step(actions)

            for log in step_logs:
                manager._log_agent_step(
                    agent_id=log['agent_id'],
                    episode=e,
                    step=step,
                    action=log['action'],
                    info=log['pre_info'],
                    next_info=next_info[f'agent_{log["agent_id"]}'],
                    env=env
                )

            step += 1
            obs = next_obs

            logger.info(f"Episode {e} - {env.render()}")

            if any(dones.values()) or any(truncations.values()):
                rendered_data = env.render()
                episode_observed_states.append(env.observed_state)

                current_episode_agent_positions = {}
                for agent_name in env.agents:
                    position = env.agents_data[agent_name]['position']
                    scaling_factor = env.agents_data[agent_name]['scaling_factor']
                    if position not in current_episode_agent_positions:
                        current_episode_agent_positions[position] = []
                    current_episode_agent_positions[position].append(f"{agent_name} scaling factor: {scaling_factor}")
                last_episode_agent_positions = current_episode_agent_positions

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

    llm_logs = format_logs_for_llm(manager.logs, env, agent_id=0)
    print(llm_logs)

    yandex_explain(logs=llm_logs, agent_id=0)
    gigachat_explain(logs=llm_logs, agent_id=0)
