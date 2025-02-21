import itertools
import numpy as np
import torch
from ppo.scheduler import SurgeryQuotaScheduler
from mappo import MAPPOTrainer, MAPPOTester

rng = np.random.default_rng(None)


class IndependentHumanEvaluator:
    def __init__(self, id, approval_rate):
        self.id = id
        self.approval_rate = approval_rate

    def evaluate(self):
        return rng.choice([True, False], p=[self.approval_rate, 1 - self.approval_rate])


class MultiAgentSystemOperator:
    def __init__(self, preferences={day: {'min': 1, 'max': 3} for day in range(7)}, max_tries=3, tau=0.05):
        self.preferences = preferences
        self.max_tries = max_tries
        self.current_try = 0
        self.tau = tau

    def setup_agents(self, agents):
        self.agents = {agent: {'active': True,
                               'base_reward': 1.0,
                               'window': 3,
                               'alpha': 2.0,
                               'urgency': urgency,
                               'complexity': complexity,
                               'completeness': completeness,
                               'mutation_rate': 0.0}
                       for (urgency, complexity, completeness), agent
                       in zip(itertools.product(range(1, 4), range(0, 2), range(0, 2)), agents)}

    def tune_agents(self, feedback):
        # if self.current_try != 0:
        #     for agent_id, agent_data in self.agents.items():
        #         if agent_data['active'] is True and feedback[agent_id]:
        #             agent_data['active'] = False
        #         elif agent_data['active'] is True and not feedback[agent_id]:
        #             agent_data['base_reward'] += (-1) ** rng.integers(1, 3) * rng.integers(1, 11) / 100
        #             agent_data['window'] += 2
        #             agent_data['alpha'] -= 0.2
        self.current_try += 1
        return self.agents


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_model = 'trained_model'
    finetuned_model = 'finetuned_model'

    humans = [IndependentHumanEvaluator(id=f'agent_{idx}',
                                        approval_rate=rng.choice([p / 100 for p in range(35, 75, 5)]))
              for idx in range(21)]

    operator = MultiAgentSystemOperator(
        preferences={day: {'min': np.min(numbers),
                           'max': np.max(numbers)} for day, numbers in
                     enumerate(rng.integers(1, 3, (7, 2)))},
        tau=0.15
    )
    operator.setup_agents([human.id for human in humans])

    agents = operator.agents
    preferences = operator.preferences

    env = SurgeryQuotaScheduler(render_mode='terminal', max_agents=12,
                                max_days=7, max_episode_length=3)
    env.reset()

    num_agents = env.num_agents
    obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").n

    print('Just in Time Preference Optimization (JITPO) concept proof simulation')
    print('Multi-Agent System operator preferences:', preferences)
    print('Propose initial draft of the schedule')

    tester = MAPPOTester(env, n_agents=num_agents, obs_dim=obs_dim,
                         action_dim=action_dim, options={'agents_data': agents,
                                                         'target_state': preferences,
                                                         'expert_logic': True})
    tester.load_model(pretrained_model)
    delta = tester.bootstrap_test(n_episodes=5, max_steps=3,
                                  target_state=preferences)

    while delta < operator.tau and operator.current_try <= operator.max_tries:
        agents = operator.tune_agents(feedback={human.id: human.evaluate() for human in humans})

        trainer = MAPPOTrainer(env, n_agents=num_agents, obs_dim=obs_dim,
                               action_dim=action_dim, options={'agents_data': agents,
                                                               'target_state': preferences,
                                                               'expert_logic': True})
        trainer.load_model(pretrained_model)
        trainer.train(n_episodes=2000, max_steps=3, log_interval=100)
        trainer.save_model(finetuned_model)

        tester = MAPPOTester(env, n_agents=num_agents, obs_dim=obs_dim,
                             action_dim=action_dim, options={'agents_data': agents,
                                                             'target_state': preferences,
                                                             'expert_logic': True})
        tester.load_model(finetuned_model)
        delta = tester.bootstrap_test(n_episodes=5, max_steps=3,
                                      target_state=preferences)

    env.close()
