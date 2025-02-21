import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.stats import levy_stable


class GeneticAlgorithmScheduler:
    def __init__(self, max_agents=12, max_days=7, population_size=100, generations=1000):
        self.max_agents = max_agents
        self.max_days = max_days
        self.population_size = population_size
        self.generations = generations
        self.agents_data = self.generate_agents_data()
        self.target_state = {day: self.max_agents / self.max_days for day in range(self.max_days)}

        self.log_dir = 'logs/genetic_algorithm'
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def generate_agents_data(self):
        agents_data = {}
        for i in range(self.max_agents):
            urgency = np.random.randint(1, 4)
            completeness = np.random.randint(0, 2)
            complexity = np.random.randint(0, 2)
            scaling_factor = ((complexity + (1 - completeness)) * urgency)
            agents_data[f"agent_{i}"] = {
                'urgency': urgency,
                'completeness': completeness,
                'complexity': complexity,
                'scaling_factor': scaling_factor,
                'base_reward': 1.0,
                'window': 3,
                'alpha': 2.0
            }
        return agents_data

    def create_individual(self):
        return np.random.randint(0, self.max_days, self.max_agents)

    def create_population(self):
        return [self.create_individual() for _ in range(self.population_size)]

    def fitness(self, individual):
        observed_state = {day: 0 for day in range(self.max_days)}
        for agent, day in enumerate(individual):
            observed_state[day] += 1

        fitness_score = 0
        for agent, day in enumerate(individual):
            agent_data = self.agents_data[f"agent_{agent}"]
            discrepancy = {d: abs(observed_state[d] - self.target_state[d]) for d in range(self.max_days)}
            window = [day + np.ceil(d - agent_data['window'] / 2) for d in range(agent_data['window'])]
            window = [int(d % self.max_days) for d in window]
            masked_discrepancy = {d: discrepancy[d] if d in window else 0 for d in discrepancy}
            rv = levy_stable(agent_data['alpha'], 0.0, loc=day, scale=1.0)
            weighted_discrepancy_sum = np.sum([rv.pdf(d) * masked_discrepancy[d] for d in masked_discrepancy])
            reward = - weighted_discrepancy_sum * agent_data['base_reward'] / max(1, agent_data['scaling_factor'])
            fitness_score += reward

        return fitness_score

    def selection(self, population, k=2):
        selected = np.random.choice(len(population), k, replace=False)
        return max([population[i] for i in selected], key=self.fitness)

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1))
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def mutation(self, individual, mutation_rate=0.01):
        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                individual[i] = np.random.randint(0, self.max_days)
        return individual

    def run(self):
        population = self.create_population()

        for generation in tqdm(range(self.generations)):
            new_population = []

            for _ in range(self.population_size):
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                new_population.append(child)

            population = new_population

            best_individual = max(population, key=self.fitness)
            best_fitness = self.fitness(best_individual)

            with self.summary_writer.as_default():
                tf.summary.scalar('Best Fitness', best_fitness, step=generation)

        return max(population, key=self.fitness)

    def get_schedule_details(self, best_individual):
        schedule_details = []
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for agent, day in enumerate(best_individual):
            agent_data = self.agents_data[f"agent_{agent}"]
            schedule_details.append({
                'agent': f"agent_{agent}",
                'day': days[day],
                'urgency': agent_data['urgency'],
                'completeness': agent_data['completeness'],
                'complexity': agent_data['complexity'],
                'scaling_factor': agent_data['scaling_factor']
            })

        return schedule_details


scheduler = GeneticAlgorithmScheduler()
best_schedule = scheduler.run()
schedule_details = scheduler.get_schedule_details(best_schedule)

for item in schedule_details:
    print(f"{item['agent']} placed on {item['day']}, urgency: {item['urgency']}, "
          f"completeness: {item['completeness']}, complexity: {item['complexity']}, "
          f"scaling_factor: {item['scaling_factor']}")
