import numpy as np
import itertools

class Client:
    def __init__(self, name, urgency, completeness, complexity):
        self.name = name
        self.urgency = urgency
        self.completeness = completeness
        self.complexity = complexity

    @property
    def scaling_factor(self):
        return (self.complexity + (1 - self.completeness)) * self.urgency
    
class Manager:
    def __init__(self, planning_horizon):
        self.planning_horizon = planning_horizon
        self.preferences = {
            day: {'min': np.min(numbers), 'max': np.max(numbers)}
            for day, numbers in enumerate(np.random.randint(1, 3, (7, 2)))
        }

    def reset_shedule(self):
        self.schedule = {day: [] for day in range(self.planning_horizon)}

    def manage(self, upcoming_requests):
        for request in upcoming_requests:
            for day in range(self.planning_horizon):
                if len(self.schedule[day]) < self.preferences[day]['max']:
                    self.schedule[day].append(request)
                    break
        return self.schedule

class Estimator():
    def __init__(self, shedules, target_state=None):
        self.observed_states = [{day: len(shedule[day]) for day in shedule.keys()} for shedule in shedules]
        self.shedules = shedules
        if target_state is not None:
            self.target_state = target_state
        else:
            self.target_state = {day: {'min': np.min(numbers), 'max': np.max(numbers)}
                                for day, numbers in enumerate(np.random.random_integers(1, 3, (7, 2)))}

    def describe(self):
        num_episodes = len(self.observed_states)

        wating_time = []
        for shedule in self.shedules:
            for day in shedule:
                for client in shedule[day]:
                    if client.scaling_factor > 3:
                        wating_time.append(day)
        print(np.mean(wating_time))

        bootstrap_average_distribution = {i: sum(episode[i] for episode in self.observed_states) / num_episodes for i in
                                          range(7)}
        print(bootstrap_average_distribution)
        bootstrap_average_percentage_deviations = {i: 0 for i in range(7)}
        for key, value in bootstrap_average_distribution.items():
            if value <= self.target_state[key]['min']:
                bootstrap_average_percentage_deviations[key] = (self.target_state[key]['min'] - value) / self.target_state[key][
                    'min']
            elif value >= self.target_state[key]['max']:
                bootstrap_average_percentage_deviations[key] = (value - self.target_state[key]['max']) / self.target_state[key][
                    'max']
            else:
                bootstrap_average_percentage_deviations[key] = 0.0

        average_bootstrap_deviation = np.mean(list(bootstrap_average_percentage_deviations.values()))
        print("Average deviation:", average_bootstrap_deviation)


if __name__ == '__main__':
    health_state = itertools.product(range(1, 4),
                                     range(0, 2),
                                     range(0, 2))
    manager = Manager(planning_horizon=7)
    print(manager.preferences)
    shedules = []
    for _ in range(1):
        manager.reset_shedule()
        shedules.append(manager.manage(np.random.choice([Client(name=f'client_{i}',
                                                                urgency=urgency,
                                                                completeness=completeness,
                                                                complexity=complexity) for i, (urgency,
                                                                                               completeness,
                                                                                               complexity)
                                                                                               in enumerate(health_state)],
                                                                                               size=12, replace=False)))
    eval = Estimator(shedules)
    eval.describe()
