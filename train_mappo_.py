import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from scheduler_ import SurgeryQuotaScheduler
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm


# Define the Actor network
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


# Define the Critic network
class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        return self.network(obs)


# MAPPO Agent
class MAPPOAgent:
    def __init__(self, obs_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, epsilon=0.2):
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, obs):
        obs = torch.FloatTensor(obs)
        probs = self.actor(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, obs, actions, old_log_probs, rewards, next_obs, dones):
        obs = torch.FloatTensor(obs)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        next_obs = torch.FloatTensor(next_obs)
        dones = torch.FloatTensor(dones)

        # Compute advantage
        values = self.critic(obs).squeeze()
        next_values = self.critic(next_obs).squeeze()
        td_target = rewards + self.gamma * next_values * (1 - dones)
        td_error = td_target - values
        advantage = td_error.detach()

        # Update critic
        critic_loss = (td_error ** 2).mean()
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update actor
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


# MAPPO Trainer
class MAPPOTrainer:
    def __init__(self, env, n_agents, obs_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, epsilon=0.2):
        self.env = env
        self.n_agents = n_agents
        self.agents = [MAPPOAgent(obs_dim, action_dim, lr_actor, lr_critic, gamma, epsilon) for _ in range(n_agents)]
        self.writer = SummaryWriter()

    def train(self, n_episodes, max_steps, log_interval=5):
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

                # Store episode rewards
                for i in range(self.n_agents):
                    episode_rewards[i] += rewards[f"agent_{i}"]

                # Update agents
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

            # Log metrics every log_interval episodes
            if (episode + 1) % log_interval == 0:
                avg_reward = sum(episode_rewards) / self.n_agents
                avg_critic_loss = np.mean(episode_critic_losses)
                avg_actor_loss = np.mean(episode_actor_losses)
                self.writer.add_scalar('Reward/average', avg_reward, episode)
                self.writer.add_scalar('Loss/critic', avg_critic_loss, episode)
                self.writer.add_scalar('Loss/actor', avg_actor_loss, episode)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'avg_reward': f'{sum(episode_rewards) / self.n_agents:.2f}'})

        pbar.close()

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), os.path.join(path, f'actor_{i}.pth'))
            torch.save(agent.critic.state_dict(), os.path.join(path, f'critic_{i}.pth'))


# Main training loop
if __name__ == "__main__":
    env = SurgeryQuotaScheduler()
    n_agents = 12
    obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").n

    trainer = MAPPOTrainer(env, n_agents, obs_dim, action_dim)
    trainer.train(n_episodes=12000, max_steps=7, log_interval=10)
    trainer.save_model('trained_model')
