import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from scheduler import ResourceScheduler
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm


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

class Critic(nn.Module):
    """
    A neural network model representing the Critic in a reinforcement learning setup.

    This class defines the architecture of the Critic model, which is used to estimate
    the value of a given observation.

    Attributes:
        network (nn.Sequential): The neural network architecture consisting of linear layers and ReLU activations.
    """
    def __init__(self, obs_dim):
        """
        Initializes the Critic model with specified observation dimension.

        Args:
            obs_dim (int): The dimension of the observation space.
        """
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        """
        Performs a forward pass through the network to compute the value of the observation.

        Args:
            obs (torch.Tensor): The input observation tensor.

        Returns:
            torch.Tensor: The output tensor representing the value of the observation.
        """
        return self.network(obs)

class MAPPOAgent:
    """
    An agent that uses the Multi-Agent Proximal Policy Optimization (MAPPO) algorithm.

    This class manages the Actor and Critic models, and handles the training process
    including action selection, updating the models, and logging.

    Attributes:
        actor (Actor): The Actor model used to select actions.
        critic (Critic): The Critic model used to estimate the value of observations.
        optimizer_actor (optim.Adam): The optimizer for the Actor model.
        optimizer_critic (optim.Adam): The optimizer for the Critic model.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The clipping parameter for the PPO objective.
    """
    def __init__(self, obs_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, epsilon=0.2):
        """
        Initializes the MAPPOAgent with specified observation and action dimensions.

        Args:
            obs_dim (int): The dimension of the observation space.
            action_dim (int): The dimension of the action space.
            lr_actor (float, optional): Learning rate for the Actor model. Defaults to 3e-4.
            lr_critic (float, optional): Learning rate for the Critic model. Defaults to 3e-4.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.99.
            epsilon (float, optional): The clipping parameter for the PPO objective. Defaults to 0.2.
        """
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, obs):
        """
        Selects an action based on the given observation using the Actor model.

        Args:
            obs (np.ndarray): The observation array.

        Returns:
            tuple: A tuple containing the selected action and its log probability.
        """
        obs = torch.FloatTensor(obs)
        probs = self.actor(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, obs, actions, old_log_probs, rewards, next_obs, dones):
        """
        Updates the Actor and Critic models based on the given batch of experiences.

        Args:
            obs (np.ndarray): The observation array.
            actions (list): The list of actions taken.
            old_log_probs (list): The list of log probabilities of the actions taken.
            rewards (list): The list of rewards received.
            next_obs (np.ndarray): The next observation array.
            dones (list): The list indicating whether the episode is done.

        Returns:
            tuple: A tuple containing the critic loss and actor loss.
        """
        obs = torch.FloatTensor(obs)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        next_obs = torch.FloatTensor(next_obs)
        dones = torch.FloatTensor(dones)

        values = self.critic(obs).squeeze()
        next_values = self.critic(next_obs).squeeze()
        td_target = rewards + self.gamma * next_values * (1 - dones)
        td_error = td_target - values
        advantage = td_error.detach()

        critic_loss = (td_error ** 2).mean()
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        probs = self.actor(obs)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        return critic_loss.item(), actor_loss.item()


class MAPPOTrainer:
    """
    A trainer class for the Multi-Agent Proximal Policy Optimization (MAPPO) algorithm.

    This class manages the training process for multiple agents in a shared environment,
    including running episodes, updating the models, and logging the training progress.

    Attributes:
        env (ResourceScheduler): The environment in which the agents operate.
        n_agents (int): The number of agents.
        agents (list): A list of MAPPOAgent instances.
        writer (SummaryWriter): A TensorBoard SummaryWriter for logging training progress.
    """
    def __init__(self, env, n_agents, obs_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, epsilon=0.2):
        """
        Initializes the MAPPOTrainer with the specified environment and agent parameters.

        Args:
            env (ResourceScheduler): The environment in which the agents operate.
            n_agents (int): The number of agents.
            obs_dim (int): The dimension of the observation space.
            action_dim (int): The dimension of the action space.
            lr_actor (float, optional): Learning rate for the Actor model. Defaults to 3e-4.
            lr_critic (float, optional): Learning rate for the Critic model. Defaults to 3e-4.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.99.
            epsilon (float, optional): The clipping parameter for the PPO objective. Defaults to 0.2.
        """
        self.env = env
        self.n_agents = n_agents
        self.agents = [MAPPOAgent(obs_dim, action_dim, lr_actor, lr_critic, gamma, epsilon) for _ in range(n_agents)]
        self.writer = SummaryWriter()

    def train(self, n_episodes, max_steps, log_interval=5):
        """
        Trains the agents for a specified number of episodes.

        Args:
            n_episodes (int): The number of episodes to train.
            max_steps (int): The maximum number of steps per episode.
            log_interval (int, optional): The interval at which to log training progress. Defaults to 5.
        """
        pbar = tqdm(total=n_episodes, desc="Training Progress")
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_rewards = [0 for _ in range(self.n_agents)]
            episode_critic_losses = []
            episode_actor_losses = []

            for step in range(max_steps):
                actions = []
                old_log_probs = []

                for i, agent in enumerate(self.agents):
                    action, log_prob = agent.get_action(obs[f"agent_{i}"])
                    actions.append(action)
                    old_log_probs.append(log_prob)

                next_obs, rewards, dones, truncations, _ = self.env.step(
                    {f"agent_{i}": a for i, a in enumerate(actions)})

                for i in range(self.n_agents):
                    episode_rewards[i] += rewards[f"agent_{i}"]

                for i, agent in enumerate(self.agents):
                    critic_loss, actor_loss = agent.update(
                        obs[f"agent_{i}"].reshape(1, -1),
                        [actions[i]],
                        [old_log_probs[i].item()],
                        [rewards[f"agent_{i}"]],
                        next_obs[f"agent_{i}"].reshape(1, -1),
                        [dones[f"agent_{i}"] or truncations[f"agent_{i}"]]
                    )
                    episode_critic_losses.append(critic_loss)
                    episode_actor_losses.append(actor_loss)

                obs = next_obs

                if all(dones.values()) or all(truncations.values()):
                    break

            if (episode + 1) % log_interval == 0:
                avg_reward = sum(episode_rewards) / self.n_agents
                avg_critic_loss = np.mean(episode_critic_losses)
                avg_actor_loss = np.mean(episode_actor_losses)
                self.writer.add_scalar('Reward/average', avg_reward, episode)
                self.writer.add_scalar('Loss/critic', avg_critic_loss, episode)
                self.writer.add_scalar('Loss/actor', avg_actor_loss, episode)

            pbar.update(1)
            pbar.set_postfix({'avg_reward': f'{sum(episode_rewards) / self.n_agents:.2f}'})

        pbar.close()

    def save_model(self, path):
        """
        Saves the Actor and Critic models for all agents to the specified path.

        Args:
            path (str): The path to the directory where the models should be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), os.path.join(path, f'actor_{i}.pth'))
            torch.save(agent.critic.state_dict(), os.path.join(path, f'critic_{i}.pth'))


if __name__ == "__main__":
    env = ResourceScheduler()
    n_agents = 12
    obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").n

    trainer = MAPPOTrainer(env, n_agents, obs_dim, action_dim)
    trainer.train(n_episodes=12000, max_steps=7, log_interval=10)
    trainer.save_model('trained_model')
