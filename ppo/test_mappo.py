import torch
import torch.nn as nn
from torch.distributions import Categorical
from ppo.scheduler import SurgeryQuotaScheduler


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


# MAPPO Agent for testing
class MAPPOAgent:
    def __init__(self, obs_dim, action_dim):
        self.actor = Actor(obs_dim, action_dim)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

        # self.c

    def get_action(self, obs):
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            probs = self.actor(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()


# MAPPO Tester
class MAPPOTester:
    def __init__(self, env, n_agents, obs_dim, action_dim, model_path):
        self.env = env
        self.n_agents = n_agents
        self.agents = [MAPPOAgent(obs_dim, action_dim) for _ in range(n_agents)]
        for i, agent in enumerate(self.agents):
            agent.load_model(f'{model_path}/actor_{i}.pth')

    def test(self, n_episodes, max_steps):
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_rewards = [0 for _ in range(self.n_agents)]

            for step in range(max_steps):
                actions = {}

                for i, agent in enumerate(self.agents):
                    action = agent.get_action(obs[f"agent_{i}"])
                    actions[f"agent_{i}"] = action

                next_obs, rewards, dones, truncations, _ = self.env.step(actions)

                # Store episode rewards
                for i in range(self.n_agents):
                    episode_rewards[i] += rewards[f"agent_{i}"]

                # Output rewards and environment state
                print(f"Step {step + 1}:")
                print("Rewards:", rewards)
                print("Environment State:")
                print(self.env.render())
                print("\n")
                print(self.env.reset())

                obs = next_obs

                if all(dones.values()) or all(truncations.values()):
                    break

            print(f"Episode {episode + 1} finished. Total rewards:", episode_rewards)
            print(f"Average reward: {sum(episode_rewards) / self.n_agents}")
            print("\n" + "=" * 50 + "\n")


# Main testing loop
if __name__ == "__main__":
    env = SurgeryQuotaScheduler(render_mode='terminal')
    n_agents = 12
    obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").n

    tester = MAPPOTester(env, n_agents, obs_dim, action_dim, '../trained_model')
    tester.test(n_episodes=5, max_steps=7)
