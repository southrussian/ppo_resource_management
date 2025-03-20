import itertools
import numpy as np
import torch
from scheduler_ import ResourceScheduler
from mappo import MAPPOTrainer, MAPPOTester


planning_horizon = 7


class IndependentHumanEvaluator:
    def __init__(self, id, approval_rate):
        self.id = id
        self.approval_rate = approval_rate

    def evaluate(self):
        return np.random.choice([True, False],
                                p=[self.approval_rate, 1 - self.approval_rate])


class MultiAgentSystemOperator:
    def __init__(self, preferences):
        self.preferences = preferences
        self.current_try = 0

    def setup_agents(self, agents=None):
        if agents is not None:
            self.agents = agents
        else:
            health_state = list(itertools.product(range(1, 4),
                                             range(0, 2),
                                             range(0, 2)))
            agents = [f'agent_{i}' for i in range(len([state for state in health_state]))]
            self.agents = {agent: {'active': True,
                                   'base_reward': 1.0,
                                   'window': 3,
                                   'alpha': 2.0,
                                   'urgency': urgency,
                                   'complexity': complexity,
                                   'completeness': completeness,
                                   'mutation_rate': 0.0}
                                   for (urgency, complexity, completeness), agent
                                   in zip(health_state, agents)}

    def tune_agents(self, feedback):
        for agent_id, agent_data in self.agents.items():
            if agent_data['active'] is True and feedback[agent_id]:
                agent_data['active'] = False
        return self.agents

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_model = 'trained_model'
    finetuned_model = 'finetuned_model'

    operator = MultiAgentSystemOperator(
        preferences={day: {'min': np.min(numbers),
                           'max': np.max(numbers)} for day, numbers in
                     enumerate(np.random.random_integers(1, 3,
                                                  (planning_horizon, 2)))}
    )

    operator.setup_agents()

    agents = operator.agents
    preferences = operator.preferences

    env = ResourceScheduler(render_mode='terminal', max_agents=12,
                            max_days=planning_horizon, max_episode_length=3)
    env.reset()

    num_agents = env.num_agents
    obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").n

    print('Preference Optimization simulation')

    print('The operator of a multi-agent system defines his preferences:', preferences)

    print('Propose initial draft of the schedule and test the algorithms performance against the given preferences')

    tester = MAPPOTester(env, n_agents=num_agents, obs_dim=obs_dim,
                         action_dim=action_dim, options={'agents_data': agents,
                                                         'target_state': preferences,
                                                         'expert_logic': True,
                                                         'priority_rate': 0.7})

    tester.load_model(pretrained_model)
    delta = tester.bootstrap_test(n_episodes=5, max_steps=3,
                                  target_state=preferences)

    print('Fine-tune the algorithm for new preferences')

    trainer = MAPPOTrainer(env, n_agents=num_agents, obs_dim=obs_dim,
                            action_dim=action_dim, options={'agents_data': agents,
                                                            'target_state': preferences,
                                                            'expert_logic': True,
                                                            'priority_rate': 0.7})
    trainer.load_model(pretrained_model)
    trainer.train(n_episodes=2000, max_steps=3, log_interval=500)
    trainer.save_model(finetuned_model)

    print('Test the fine-tuned algorithms performance against the given preferences')

    tester = MAPPOTester(env, n_agents=num_agents, obs_dim=obs_dim,
                            action_dim=action_dim, options={'agents_data': agents,
                                                            'target_state': preferences,
                                                            'expert_logic': True,
                                                            'priority_rate': 0.7})
    tester.load_model(finetuned_model)
    delta = tester.bootstrap_test(n_episodes=5, max_steps=3,
                                    target_state=preferences)

    env.close()

    print('Start working with clients')

    humans = [
        IndependentHumanEvaluator(id=f'agent_{idx}',
                                  approval_rate= \
                                    np.random.choice(
                                        [p / 100 for p in range(35, 75, 5)]
                                    )
                                    )
              for idx in range(100)
            ]

    # Реализовать цикл обработки 100 пациентов полученными моделями