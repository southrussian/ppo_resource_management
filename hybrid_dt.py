import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
from scheduler_ import SurgeryQuotaScheduler

# Определение кастомной модели Decision Transformer
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, max_length, n_layers=3, n_heads=8, embed_dim=128):
        super(DecisionTransformer, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length

        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.action_embedding = nn.Embedding(action_dim, embed_dim)
        self.reward_embedding = nn.Linear(1, embed_dim)
        self.timestep_embedding = nn.Embedding(max_length, embed_dim)

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )

        self.action_head = nn.Linear(embed_dim, action_dim)
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, states, actions, rewards, timesteps, attention_mask=None):
        batch_size = states.size(0)
        seq_length = states.size(1)

        states = self.state_embedding(states)
        actions = self.action_embedding(actions)
        rewards = self.reward_embedding(rewards.unsqueeze(-1)).squeeze(-1)
        timesteps = self.timestep_embedding(timesteps)

        input_embedding = states + actions + rewards + timesteps

        transformer_output = self.transformer(input_embedding, input_embedding, src_key_padding_mask=attention_mask)

        action_logits = self.action_head(transformer_output)
        values = self.value_head(transformer_output)

        return action_logits, values

# Обновление MAPPOAgent для использования Decision Transformer
class MAPPOAgent:
    def __init__(self, obs_dim, action_dim, max_length, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, epsilon=0.2):
        self.dt = DecisionTransformer(obs_dim, action_dim, max_length)
        self.optimizer = optim.Adam(self.dt.parameters(), lr=lr_actor)
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_length = max_length

    def get_action(self, obs, actions, rewards, timesteps):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        actions = torch.LongTensor(actions).unsqueeze(0)
        rewards = torch.FloatTensor(rewards).unsqueeze(0)
        timesteps = torch.LongTensor(timesteps).unsqueeze(0)
        action_logits, _ = self.dt(obs, actions, rewards, timesteps)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        if action.numel() > 0:
            return action.item(), dist.log_prob(action)
        else:
            return None, None

    def update(self, obs, actions, rewards, timesteps, old_log_probs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        actions = torch.LongTensor(actions).unsqueeze(0)
        rewards = torch.FloatTensor(rewards).unsqueeze(0)
        timesteps = torch.LongTensor(timesteps).unsqueeze(0)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(0)

        action_logits, values = self.dt(obs, actions, rewards, timesteps)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions)

        # Compute advantage
        next_values = values[:, -1]
        td_target = rewards + self.gamma * next_values
        td_error = td_target - values
        advantage = td_error.detach()

        # Update policy
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Update value function
        value_loss = (td_error ** 2).mean()

        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item()

# Обновление MAPPOTrainer для использования Decision Transformer
# Обновление MAPPOTrainer для использования индивидуальных историй агентов
class MAPPOTrainer:
    def __init__(self, env, n_agents, obs_dim, action_dim, max_length, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99,
                 epsilon=0.2):
        self.env = env
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.agents = [MAPPOAgent(obs_dim, action_dim, max_length, lr_actor, lr_critic, gamma, epsilon) for _ in
                       range(n_agents)]
        self.writer = SummaryWriter()
        self.max_length = max_length  # Добавляем max_length как атрибут

    def train(self, n_episodes, max_steps, log_interval=5):
        pbar = tqdm(total=n_episodes, desc="Training Progress")
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_rewards = [0 for _ in range(self.n_agents)]
            episode_policy_losses = []
            episode_value_losses = []

            # Инициализация истории для каждого агента
            agent_histories = [{
                'states': [],
                'actions': [],
                'rewards': [],
                'timesteps': [],
            } for _ in range(self.n_agents)]

            for step in range(max_steps):
                actions = []
                old_log_probs = []
                current_rewards = []
                current_timesteps = []

                # Сбор действий для каждого агента с учетом их истории
                for i, agent in enumerate(self.agents):
                    history = agent_histories[i]
                    # Подготовка данных для Decision Transformer
                    states = torch.FloatTensor(history['states'][-self.max_length:]).unsqueeze(0) if history[
                        'states'] else torch.zeros((1, 0, self.obs_dim))
                    acts = torch.LongTensor(history['actions'][-self.max_length:]).unsqueeze(0) if history[
                        'actions'] else torch.zeros((1, 0), dtype=torch.long)
                    rews = torch.FloatTensor(history['rewards'][-self.max_length:]).unsqueeze(0) if history[
                        'rewards'] else torch.zeros((1, 0, 1))
                    times = torch.LongTensor(history['timesteps'][-self.max_length:]).unsqueeze(0) if history[
                        'timesteps'] else torch.zeros((1, 0), dtype=torch.long)

                    # Получение текущего состояния
                    current_state = torch.FloatTensor(obs[f"agent_{i}"]).unsqueeze(0).unsqueeze(0)

                    # Объединение истории с текущим состоянием
                    all_states = torch.cat([states, current_state], dim=1) if states.size(1) > 0 else current_state
                    all_actions = acts
                    all_rewards = rews
                    all_timesteps = times

                    # Получение действия от агента
                    action, log_prob = agent.get_action(all_states, all_actions, all_rewards, all_timesteps)
                    if action is not None:
                        actions.append(action)
                        old_log_probs.append(log_prob)
                        current_rewards.append(0)  # Временная заглушка для наград
                        current_timesteps.append(step)
                    else:
                        # Обработка случая, если действие не получено
                        actions.append(0)  # Действие по умолчанию
                        old_log_probs.append(torch.tensor(0.0))  # Заглушка для лога вероятности

                # Выполнение шага в среде
                next_obs, rewards_dict, dones, truncations, _ = self.env.step(
                    {f"agent_{i}": actions[i] for i in range(len(actions))}
                )

                # Обновление истории для каждого агента
                for i in range(self.n_agents):
                    agent_hist = agent_histories[i]
                    agent_hist['states'].append(obs[f"agent_{i}"])
                    agent_hist['actions'].append(actions[i])
                    agent_hist['rewards'].append(rewards_dict.get(f"agent_{i}", 0))
                    agent_hist['timesteps'].append(step)
                    # Обрезаем историю до max_length
                    if len(agent_hist['states']) > self.max_length:
                        agent_hist['states'] = agent_hist['states'][-self.max_length:]
                        agent_hist['actions'] = agent_hist['actions'][-self.max_length:]
                        agent_hist['rewards'] = agent_hist['rewards'][-self.max_length:]
                        agent_hist['timesteps'] = agent_hist['timesteps'][-self.max_length:]

                # Обновление агентов
                for i, agent in enumerate(self.agents):
                    history = agent_histories[i]
                    if len(history['states']) == 0:
                        continue  # Пропустить, если история пуста

                    states = torch.FloatTensor(history['states']).unsqueeze(0)
                    actions_tensor = torch.LongTensor(history['actions']).unsqueeze(0)
                    rewards_tensor = torch.FloatTensor(history['rewards']).unsqueeze(0)
                    timesteps_tensor = torch.LongTensor(history['timesteps']).unsqueeze(0)

                    policy_loss, value_loss = agent.update(
                        states,
                        actions_tensor,
                        rewards_tensor,
                        timesteps_tensor,
                        torch.stack(old_log_probs[i:i + 1])  # Исправляем получение old_log_probs
                    )
                    episode_policy_losses.append(policy_loss)
                    episode_value_losses.append(value_loss)

                # Обновление наблюдений и наград
                obs = next_obs
                for i in range(self.n_agents):
                    episode_rewards[i] += rewards_dict.get(f"agent_{i}", 0)

                if all(dones.values()) or all(truncations.values()):
                    break

            # Логирование и обновление прогресса
            if (episode + 1) % log_interval == 0:
                avg_reward = sum(episode_rewards) / self.n_agents
                avg_policy_loss = np.mean(episode_policy_losses)
                avg_value_loss = np.mean(episode_value_losses)
                self.writer.add_scalar('Reward/average', avg_reward, episode)
                self.writer.add_scalar('Loss/policy', avg_policy_loss, episode)
                self.writer.add_scalar('Loss/value', avg_value_loss, episode)

            pbar.update(1)
            pbar.set_postfix({'avg_reward': f'{sum(episode_rewards) / self.n_agents:.2f}'})

        pbar.close()

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for i, agent in enumerate(self.agents):
            torch.save(agent.dt.state_dict(), os.path.join(path, f'dt_{i}.pth'))


# Основной цикл обучения
if __name__ == "__main__":
    env = SurgeryQuotaScheduler()
    n_agents = 12
    obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").n
    max_length = 10  # Максимальная длина последовательности для Decision Transformer

    trainer = MAPPOTrainer(env, n_agents, obs_dim, action_dim, max_length)
    trainer.train(n_episodes=12000, max_steps=7, log_interval=10)
    trainer.save_model('trained_model')
