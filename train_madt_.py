import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random
from tqdm import tqdm

from scheduler_ import SurgeryQuotaScheduler


class MultiAgentDT(nn.Module):
    def __init__(self, num_agents, state_dim, act_dim, hidden_dim, num_layers, num_heads, max_len):
        super().__init__()
        self.num_agents = num_agents
        self.act_dim = act_dim
        self.max_len = max_len

        # Энкодеры для состояний, действий и возвратов
        self.state_enc = nn.Linear(state_dim, hidden_dim)
        self.act_enc = nn.Linear(num_agents * act_dim, hidden_dim)
        self.ret_enc = nn.Linear(1, hidden_dim)

        # Позиционные эмбеддинги
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        # Трансформер
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Выходные слои
        self.act_head = nn.Linear(hidden_dim, num_agents * act_dim)

    def forward(self, states, actions, returns, timesteps):
        batch_size = states.size(0)

        # Энкодинг состояний
        state_emb = self.state_enc(states)

        # Энкодинг действий
        act_emb = self.act_enc(actions)

        # Энкодинг возвратов
        ret_emb = self.ret_enc(returns.unsqueeze(-1))

        # Суммирование эмбеддингов
        combined_emb = state_emb + act_emb + ret_emb

        # Позиционные эмбеддинги
        pos_ids = torch.arange(timesteps, device=states.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos_ids)
        combined_emb += pos_emb

        # Прохождение через трансформер
        out = self.transformer(combined_emb)

        # Предсказание действий
        act_logits = self.act_head(out[:, -1])
        return act_logits.view(batch_size, self.num_agents, self.act_dim)


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, episode):
        self.buffer.append(episode)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def collect_episode(env, max_steps=100):
    obs, _ = env.reset()
    episode = []
    total_return = 0

    for _ in range(max_steps):
        # Случайные действия для сбора данных
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        next_obs, rewards, terms, truncs, _ = env.step(actions)

        # Сохранение данных
        episode.append({
            'states': obs,
            'actions': actions,
            'rewards': rewards,
            'returns': sum(rewards.values())
        })

        total_return += sum(rewards.values())
        obs = next_obs

        if all(terms.values()) or all(truncs.values()):
            break

    # Добавление возвратов к цели
    for i in reversed(range(len(episode) - 1)):
        episode[i]['returns'] += episode[i + 1]['returns']

    return episode, total_return


class MADTDataset(Dataset):
    def __init__(self, buffer, context_len):
        self.buffer = buffer
        self.context_len = context_len

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        episode = list(self.buffer.buffer)[idx]
        states = []
        actions = []
        returns = []

        # Нормализация данных
        max_return = max([e['returns'] for e in episode])

        for i in range(len(episode)):
            # Объединение состояний всех агентов
            state_vec = np.concatenate([v for v in episode[i]['states'].values()])
            states.append(state_vec)

            # Действия в one-hot
            act_vec = np.zeros((len(env.agents), 3))
            for j, agent in enumerate(env.agents):
                act_vec[j, episode[i]['actions'][agent]] = 1
            actions.append(act_vec.flatten())

            returns.append(episode[i]['returns'] / max_return)

        # Создание контекста
        states = np.array(states)  # Convert list to numpy array
        actions = np.array(actions)  # Convert list to numpy array
        returns = np.array(returns)  # Convert list to numpy array

        print(states.shape)

        # Ensure the states have the correct dimension
        # assert states.shape[1] == 252, f"Expected state dimension 252, but got {states.shape[1]}"

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        returns = torch.FloatTensor(returns)

        return {
            'states': states,
            'actions': actions,
            'returns': returns
        }


def train(model, buffer, batch_size=32, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataset = MADTDataset(buffer, context_len=20)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in loader:
            states = batch['states']
            actions = batch['actions']
            returns = batch['returns']

            # Ensure the sequences are correctly sliced
            seq_len = states.size(1) - 1
            if seq_len <= 0:
                continue

            # Обучение с учителем
            logits = model(
                states[:, :-1],
                actions[:, :-1],
                returns[:, :-1],
                timesteps=seq_len
            )

            # Преобразование logits и targets
            logits = logits.view(-1, model.act_dim)  # (batch_size 32 * num_agents 12, act_dim 3)
            targets = torch.argmax(actions[:, 1:], dim=-1).view(-1)  # (batch_size * num_agents)

            # Проверка размерностей
            print(f"Logits shape: {logits.shape}")
            print(f"Targets shape: {targets.shape}")

            # Расчет потерь
            loss = nn.CrossEntropyLoss()(logits, targets)

            # Оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


def test(model, env, num_episodes=10):
    total_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        states = []
        actions = []
        returns = []
        total_r = 0

        while True:
            # Подготовка текущего состояния
            state_vec = np.concatenate([v for v in obs.values()])

            # Предсказание действий
            if len(states) > 0:
                with torch.no_grad():
                    act_logits = model(
                        torch.FloatTensor([states]),
                        torch.FloatTensor([actions]),
                        torch.FloatTensor([returns]),
                        timesteps=len(states)
                    )
                act_probs = torch.softmax(act_logits, dim=-1)
                act = torch.argmax(act_probs, dim=-1).numpy()
            else:
                act = np.zeros((len(env.agents), 3))

            # Преобразование действий
            action_dict = {}
            for i, agent in enumerate(env.agents):
                action_dict[agent] = act[i]

            # Шаг в среде
            next_obs, rewards, terms, truncs, _ = env.step(action_dict)
            total_r += sum(rewards.values())

            # Обновление истории
            states.append(state_vec)
            actions.append(act.flatten())
            returns.append(total_r)

            obs = next_obs

            if all(terms.values()) or all(truncs.values()):
                break

        total_rewards.append(total_r)

    print(f"Average reward: {np.mean(total_rewards):.2f}")


if __name__ == "__main__":
    # Инициализация среды
    env = SurgeryQuotaScheduler()

    # Параметры модели
    num_agents = 12
    state_dim = 11 * num_agents  # Размер observation каждого агента * количество агентов
    act_dim = 3
    hidden_dim = 132
    num_layers = 4
    num_heads = 12
    max_len = 20

    # Инициализация модели
    model = MultiAgentDT(num_agents, state_dim, act_dim, hidden_dim, num_layers, num_heads, max_len)

    # Сбор данных
    buffer = ExperienceReplay(1000)
    print("Collecting data...")
    for _ in tqdm(range(100)):
        episode, _ = collect_episode(env)
        buffer.add(episode)

    # Обучение
    print("Training...")
    train(model, buffer, epochs=100)

    # Тестирование
    print("Testing...")
    test(model, env)
