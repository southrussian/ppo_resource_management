# Documentation

## Table of Contents
1. [Environment (env.py)](#environment-envpy)
   - [Overview](#environment-overview)
   - [Classes](#environment-classes)
   - [Key Functions](#environment-key-functions)
   - [Environment Dynamics](#environment-dynamics)
2. [MADDPG Training Script (maddpg_run.py)](#maddpg-training-script-maddpg_runpy)
   - [Overview](#maddpg-overview)
   - [Configuration](#maddpg-configuration)
   - [Key Components](#maddpg-key-components)
   - [Training Loop](#maddpg-training-loop)
   - [Metrics and Logging](#maddpg-metrics-and-logging)

## Environment (env.py) <a name="environment-envpy"></a>

### Overview <a name="environment-overview"></a>

The `env.py` file implements a custom environment called `SurgeryQuotaScheduler`. This environment simulates a hospital scheduling system where multiple agents (representing surgeries) need to be allocated across a 14-day period. The environment is designed as a multi-agent reinforcement learning task.

### Classes <a name="environment-classes"></a>

#### SurgeryQuotaScheduler

This is the main class that implements the `ParallelEnv` interface from PettingZoo.

**Key Attributes:**
- `render_mode`: Determines how the environment state is visualized.
- `num_moves`: Tracks the number of moves made in the current episode.
- `possible_agents`: List of all possible agents (surgeries).
- `agents`: Current active agents.
- `agent_data`: Dictionary storing data for each agent.

**Key Methods:**
- `reset()`: Resets the environment to its initial state.
- `step(actions)`: Executes one time step within the environment.
- `render()`: Renders the environment's current state.
- `close()`: Cleans up resources used by the environment.

### Key Functions <a name="environment-key-functions"></a>

#### reward_map(slot_occupancy, position, end_of_episode, k, action, name)

Calculates the reward for an agent based on its action and the current state of the environment.

#### generate_levy_distribution_number()

Generates a number from a Lévy distribution, used to introduce rare, high-impact events in the environment.

#### draw_boxes(win, win_width, win_height, calendar)

Visualizes the current state of the scheduling calendar using Pygame.

### Environment Dynamics <a name="environment-dynamics"></a>

- The environment simulates a 14-day scheduling period.
- Each agent (surgery) has attributes: urgency, completeness, complexity, and position.
- Agents can take three actions: move forward, move backward, or stay in place.
- The environment includes a mutation mechanism that can change the attributes of surgeries over time.
- The reward function considers the occupancy of slots and the attributes of each surgery.

## MADDPG Training Script (maddpg_run.py) <a name="maddpg-training-script-maddpg_runpy"></a>

### Overview <a name="maddpg-overview"></a>

This script implements the training loop for the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm on the Surgery Quota Scheduler environment. It utilizes the AgileRL library for reinforcement learning components and implements additional features for performance tracking and optimization.

### Configuration <a name="maddpg-configuration"></a>

#### Network Configuration (NET_CONFIG)
- Architecture: CNN
- Hidden layers: [32, 32]
- Channel sizes: [32, 32]
- Kernel sizes: [3, 3]
- Stride sizes: [2, 2]
- Normalization: Enabled

#### Initial Hyperparameters (INIT_HP)
- Population size: 2
- Algorithm: MADDPG
- Batch size: 32
- Learning rates: 0.001 (both actor and critic)
- Discount factor (gamma): 0.95
- Memory size: 100,000
- Learning frequency: 100 steps
- Soft update parameter (tau): 0.01

### Key Components <a name="maddpg-key-components"></a>

1. **MultiAgentReplayBuffer**: Stores and samples experiences for all agents.
2. **TournamentSelection**: Implements tournament selection for evolutionary optimization.
3. **Mutations**: Handles mutations of hyperparameters and network architecture.
4. **PettingZooVectorizationParallelWrapper**: Wraps the environment for parallel execution.

### Training Loop <a name="maddpg-training-loop"></a>

The training loop consists of the following main steps:

1. Reset the environment and initialize agents.
2. For each agent in the population:
   a. Collect experiences by interacting with the environment.
   b. Store experiences in the replay buffer.
   c. Perform learning updates at specified intervals.
3. Evaluate the population.
4. Perform tournament selection and mutation.
5. Repeat until the maximum number of steps is reached.

### Metrics and Logging <a name="maddpg-metrics-and-logging"></a>

The script tracks and logs several metrics:

- Cumulative reward
- Average episodic returns
- Approximate KL divergence
- Population fitness

These metrics are logged using TensorBoard for easy visualization and analysis.
