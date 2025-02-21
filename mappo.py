import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from ppo.scheduler import SurgeryQuotaScheduler
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm


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

    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path, weights_only=True))
        self.critic.load_state_dict(torch.load(critic_path, weights_only=True))
        self.actor.train()
        self.critic.train()


class MAPPOTrainer:

    def __init__(self, env, n_agents, obs_dim, action_dim, options=None, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99,
                 epsilon=0.2):
        self.env = env
        self.options = options
        self.n_agents = n_agents
        self.agents = [MAPPOAgent(obs_dim, action_dim, lr_actor, lr_critic, gamma, epsilon) for _ in range(n_agents)]
        self.writer = SummaryWriter()

    def train(self, n_episodes, max_steps, log_interval):
        pbar = tqdm(total=n_episodes, desc="Training Progress")
        for episode in range(n_episodes):
            obs, _ = self.env.reset(options=self.options)
            episode_rewards = [0 for _ in range(self.n_agents)]
            episode_critic_losses = []
            episode_actor_losses = []

            for _ in range(max_steps):
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
        if not os.path.exists(path):
            os.makedirs(path)
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), os.path.join(path, f'actor_{i}.pth'))
            torch.save(agent.critic.state_dict(), os.path.join(path, f'critic_{i}.pth'))
        print(f"Saved model to {path}")

    def load_model(self, path):
        for i, agent in enumerate(self.agents):
            agent.load_model(
                os.path.join(path, f'actor_{i}.pth'),
                os.path.join(path, f'critic_{i}.pth')
            )
        print(f"Loaded model from {path}")


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

        bootstrap_average_percentage_deviations = {i: 0 for i in range(7)}
        mean_target_state = {day: (target_state[day]['max'] + target_state[day]['min']) / 2 for day in target_state}
        for key, value in bootstrap_average_distribution.items():
            bootstrap_average_percentage_deviations[key] = np.abs(mean_target_state[key] - value) / mean_target_state[
                key]

        print(bootstrap_average_percentage_deviations.values())
        print(np.mean(list(bootstrap_average_percentage_deviations.values())))
        print("Standart deviation", np.std(list(bootstrap_average_percentage_deviations.values())))
        average_bootstrap_deviation = np.mean(list(bootstrap_average_percentage_deviations.values()))
        return average_bootstrap_deviation

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
                    action, _ = agent.get_action(obs[f"agent_{i}"])
                    actions[f"agent_{i}"] = action

                next_obs, _, dones, truncations, info = self.env.step(actions)
                print(f'Step {step}', self.env.render())
                print([(pos, sf) for pos, sf in zip([agent_info['position'] for agent_info in info.values()],
                                                    [agent_info['scaling_factor'] for agent_info in info.values()])])

                if step == max_steps - 1:
                    episode_observed_states.append(self.env.observed_state)
                    all_final_positions.extend([agent_info['position'] for agent_info in info.values()])
                    all_scaling_factors.extend([agent_info['scaling_factor'] for agent_info in info.values()])

                obs = next_obs

                if all(dones.values()) or all(truncations.values()):
                    break

            all_observed_states.extend(episode_observed_states)

        deviation = self.calculate_deviation(all_observed_states, target_state)
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

        return deviation, avg_bids_per_day, avg_position_per_scaling_factor, std_combined_positions

    def bootstrap_test(self, n_episodes, max_steps, target_state):
        deviation, avg_bids, avg_positions, std_position_deviation_per_scaling_factor = self.test(n_episodes, max_steps, target_state)

        print(f"Mean percentage deviation: {deviation:.2%}")
        print(f"Average number of bids per day: {avg_bids}")
        print(f"Average position per scaling factor: {avg_positions}")
        print(f"std_position_deviation_per_scaling_factor: {std_position_deviation_per_scaling_factor}")

        return deviation / 100

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), os.path.join(path, f'actor_{i}.pth'))
            torch.save(agent.critic.state_dict(), os.path.join(path, f'critic_{i}.pth'))
        print(f"Saved model to {path}")

    def load_model(self, path):
        for i, agent in enumerate(self.agents):
            agent.load_model(
                os.path.join(path, f'actor_{i}.pth'),
                os.path.join(path, f'critic_{i}.pth')
            )
        print(f"Loaded model from {path}")


if __name__ == "__main__":
    env = SurgeryQuotaScheduler()
    env.reset()
    n_agents = env.num_agents
    obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").n

    trainer = MAPPOTrainer(env, n_agents, obs_dim, action_dim)

    trainer.train(n_episodes=4000, max_steps=3, log_interval=100)
    trainer.save_model("trained_model")
