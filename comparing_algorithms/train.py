import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper
from ppo.scheduler import SurgeryQuotaScheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_capacity", type=int, default=3, help='max capacity of the environment')
    parser.add_argument("--max_agents", type=int, default=21, help='max number of agents')
    parser.add_argument("--max_days", type=int, default=7, help='planning horizon')
    parser.add_argument("--max_episode_length", type=int, default=7, help='max number of game turns')
    parser.add_argument("--algorithm", type=str, default="MADDPG", help="agilerl algorithm")
    parser.add_argument("--timestamp", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help='timestamp for process tracking')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the network configuration
    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": [128, 128],  # Actor hidden size
    }

    # Define the initial hyperparameters
    INIT_HP = {
        "POPULATION_SIZE": 4,
        "ALGO": args.algorithm,  # Algorithm
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 32,  # Batch size
        "O_U_NOISE": True,  # Ornstein Uhlenbeck action noise
        "EXPL_NOISE": 0.1,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.15,  # Rate of mean reversion in OU noise
        "DT": 0.01,  # Timestep for OU noise
        "LR_ACTOR": 0.001,  # Actor learning rate
        "LR_CRITIC": 0.01,  # Critic learning rate
        "GAMMA": 0.95,  # Discount factor
        "MEMORY_SIZE": 100_000,  # Max memory buffer size
        "LEARN_STEP": 5,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
    }

    def estimate_approx_kl(old_cont_actions, new_cont_actions, epsilon=1e-8):
      # Initialize lists to store KL divergences for each agent
      approx_kls = []

      # Extract action probabilities into a list for old and new actions
      old_action_probs = list(old_cont_actions.values())
      new_action_probs = list(new_cont_actions.values())

      # Convert each agent's action probabilities to numpy array and compute KL divergence
      for old_probs, new_probs in zip(old_action_probs, new_action_probs):
          # Compute softmax along the action dimension
          avg_probs = np.mean(new_probs, axis=0)
          avg_probs = F.softmax(torch.tensor(avg_probs), dim=0).numpy()  # Ensure softmax is applied along axis 0

          agent_approx_kls = []
          for probs in old_probs:
              probs = F.softmax(torch.tensor(probs), dim=0).numpy() # Ensure softmax is applied along axis 0

              # Add epsilon to avoid numerical instability
              avg_probs += epsilon
              probs += epsilon

              # Compute KL divergence
              kl = np.sum(probs * np.log(probs / avg_probs))
              agent_approx_kls.append(kl)

          # Compute average KL divergence for the current agent
          approx_kls.append(np.mean(agent_approx_kls))

      # Compute the overall average KL divergence across all agents
      return np.mean(approx_kls)

    num_envs = 8

    # Define some environment as a parallel environment
    env = SurgeryQuotaScheduler(render_mode="terminal", max_capacity=args.max_capacity,
                                max_agents=args.max_agents, max_days=args.max_days,
                                max_episode_length=args.max_episode_length)
    env = PettingZooVectorizationParallelWrapper(env, n_envs=num_envs)
    env.reset()

    # Configure the multi-agent algo input arguments
    try:
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.action_space(agent).n for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = True
        INIT_HP["MAX_ACTION"] = None
        INIT_HP["MIN_ACTION"] = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = False
        INIT_HP["MAX_ACTION"] = [env.action_space(agent).high for agent in env.agents]
        INIT_HP["MIN_ACTION"] = [env.action_space(agent).low for agent in env.agents]

    # Not applicable to MPE environments, used when images are used for observations (Atari environments)
    if INIT_HP["CHANNELS_LAST"]:
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    # Create a population ready for evolutionary hyper-parameter optimisation
    pop = create_population(
        INIT_HP["ALGO"],
        state_dim,
        action_dim,
        one_hot,
        NET_CONFIG,
        INIT_HP,
        population_size=INIT_HP["POPULATION_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    # Configure the multi-agent replay buffer
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    # Instantiate a mutations object (used for HPO)
    mutations = Mutations(
        algo=INIT_HP["ALGO"],
        no_mutation=0.2,  # Probability of no mutation
        architecture=0.2,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        rl_hp_selection=[
            "lr",
            "learn_step",
            "batch_size",
        ],  # RL hyperparams selected for mutation
        mutation_sd=0.1,  # Mutation strength
        agent_ids=INIT_HP["AGENT_IDS"],
        arch=NET_CONFIG["arch"],
        rand_seed=np.random.seed(None),
        device=device,
    )

    # Define training loop parameters
    max_steps = 13_000  # Max steps
    learning_delay = 0  # Steps before starting learning
    evo_steps = 1000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes
    elite = pop[0]  # Assign a placeholder "elite" agent

    total_steps = 0
    cumulative_reward = 0
    episode_count = 0
    episode_returns = []
    average_episodic_rewards = []

    # Initialize metrics lists
    global_steps_list = []
    cumulative_reward_list = []
    average_episodic_returns_list = []
    approx_kl_list = []
    last_policies = {}

    log_dir = "logs"
    run_name = f"{args.algorithm}_{args.timestamp}"

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps, unit="step")
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        policies = {}
        for agent in pop:  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            episode_return = 0

            if INIT_HP["CHANNELS_LAST"]:
                state = {
                    agent_id: np.moveaxis(s, [-1], [-3])
                    for agent_id, s in state.items()
                }

            for idx_step in range(evo_steps // num_envs):
                agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
                env_defined_actions = (
                    info["env_defined_actions"]
                    if "env_defined_actions" in info.keys()
                    else None
                )

                # Get next action from agent
                cont_actions, discrete_action = agent.get_action(
                    states=state,
                    training=True,
                    agent_mask=agent_mask,
                    env_defined_actions=env_defined_actions,
                )

                if agent.discrete_actions:
                    action = discrete_action
                else:
                    action = cont_actions

                # last_policy = cont_actions
                last_policies = cont_actions

                # Act in environment
                next_state, reward, termination, truncation, info = env.step(action)

                scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
                cumulative_reward += np.sum(np.array(list(reward.values())).transpose())

                total_steps += num_envs
                steps += num_envs

                # Image processing if necessary for the environment
                if INIT_HP["CHANNELS_LAST"]:
                    next_state = {
                        agent_id: np.moveaxis(ns, [-1], [-3])
                        for agent_id, ns in next_state.items()
                    }

                # Save experiences to replay buffer
                memory.save_to_memory(
                    state,
                    cont_actions,
                    reward,
                    next_state,
                    termination,
                    is_vectorised=True,
                )

                # Learn according to learning frequency
                # Handle learn steps > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):
                        # Sample replay buffer
                        experiences = memory.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)
                # Handle num_envs > learn step; learn multiple times per step in env
                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        experiences = memory.sample(agent.batch_size)

                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)

                state = next_state

                # Calculate scores and reset noise for finished episodes
                reset_noise_indices = []
                term_array = np.array(list(termination.values())).transpose()
                trunc_array = np.array(list(truncation.values())).transpose()
                for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                    if np.any(d) or np.any(t):
                        completed_episode_scores.append(scores[idx])
                        # print("completed_episode_scores", completed_episode_scores)
                        agent.scores.append(scores[idx])
                        episode_returns.append(episode_return)
                        episode_count += 1

                        scores[idx] = 0
                        episode_return = 0
                        reset_noise_indices.append(idx)

                agent.reset_action_noise(reset_noise_indices)

            pbar.update(evo_steps // len(pop))

            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

            policies = last_policies

            avg_approx_kl = estimate_approx_kl(policies, last_policies)
            approx_kl_list.append(avg_approx_kl)

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                swap_channels=INIT_HP["CHANNELS_LAST"],
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        # Log metrics
        fitnesses_dict = {f"Agent {i} Fitness": fitness for i, fitness in enumerate(fitnesses)}
        scores_dict = {f"Agent {i} Score": score for i, score in enumerate(mean_scores)}

        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
        print(f"Scores: {mean_scores}")
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}')

        # Append metrics to lists
        cumulative_reward_list.append(cumulative_reward)

        # Log metrics to TensorBoard
        writer.add_scalar("Cumulative Reward", np.mean(cumulative_reward_list), total_steps / INIT_HP['POPULATION_SIZE'])
        writer.add_scalar("Average Approx KL", np.mean(approx_kl_list), total_steps / INIT_HP['POPULATION_SIZE'])
        writer.add_scalar("Average Scores", np.mean(mean_scores), total_steps / INIT_HP['POPULATION_SIZE'])
        writer.add_scalar("Avarage Fitnesses", np.mean(fitnesses), total_steps / INIT_HP['POPULATION_SIZE'])

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    # Save the trained algorithm
    path = f"./models/{args.algorithm}_{args.timestamp}"
    filename = f"trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    elite.save_checkpoint(save_path)

    pbar.close()
    env.close()
    writer.close()
