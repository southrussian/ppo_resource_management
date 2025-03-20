from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import torch
import itertools
import logging
import matplotlib.pyplot as plt
from rescheduling_ import MAPPOAgent, Client, MultiAgentSystemOperator, ResourceScheduler, calculate_deviation, calculate_scaling_factor_positions

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('reschedule.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    num_patients = int(request.form['num_patients'])
    urgency = [int(request.form[f'urgency_{i}']) for i in range(num_patients)]
    completeness = [int(request.form[f'completeness_{i}']) for i in range(num_patients)]
    complexity = [int(request.form[f'complexity_{i}']) for i in range(num_patients)]
    initial_positions = [int(request.form[f'initial_position_{i}']) for i in range(num_patients)]
    target_state = {i: {'min': int(request.form[f'min_{i}']), 'max': int(request.form[f'max_{i}'])} for i in range(7)}

    patiens = [Client(name=f'client_{i}', urgency=urgency[i], completeness=completeness[i], complexity=complexity[i]) for i in range(num_patients)]

    urgency_range = range(1, 4)
    completeness_range = range(0, 2)
    complexity_range = range(0, 2)

    options = list(itertools.product(urgency_range, completeness_range, complexity_range))
    selected_states = np.random.choice(len(options), size=12, replace=False)
    health_states = [options[i] for i in selected_states]

    agents = {f'agent_{i}': MAPPOAgent(obs_dim=11, action_dim=3) for i in range(len(health_states))}

    for i, agent in enumerate(agents):
        agents[agent].load_model(f'trained_model/actor_{i}.pth')

    manager = MultiAgentSystemOperator(list_of_clients=patiens)
    manager.assign_agnets(
        {health_state: {'agent_name': agent_name, 'model_file': model_file} for health_state, (agent_name, model_file)
         in zip(health_states, agents.items())})

    e = 0
    all_observed_states = []
    last_episode_agent_positions = None

    while not all([client.satisfied for client in manager.clients]):
        logging.info(f"Starting episode {e}")

        env = ResourceScheduler(render_mode='terminal', max_agents=len(patiens),
                                max_days=7, max_episode_length=7)
        obs, _ = env.reset(
            options={
                'target_state': target_state,
                'agents_data': {f'agent_{i}': {'active': ~client.satisfied,
                                               'base_reward': 1.0,
                                               'window': 3,
                                               'alpha': 2.0,
                                               'urgency': client.urgency,
                                               'completeness': client.completeness,
                                               'complexity': client.complexity,
                                               'mutation_rate': 0.0}
                                for i, client in enumerate(manager.clients)},
                'initial_positions': {f'agent_{i}': initial_positions[i] for i in range(num_patients)}
            }
        )

        logging.info(f"Episode {e} - {env.render()}")

        episode_observed_states = []

        while True:
            actions = manager.get_actions(observaions=obs)
            logging.debug(f"Actions: {actions}")
            obs, _, dones, truncations, _ = env.step(actions)

            logging.info(f"Episode {e} - {env.render()}")

            if any(dones.values()) or any(truncations.values()):
                rendered_data = env.render()
                episode_observed_states.append(env.observed_state)

                current_episode_agent_positions = {}
                for agent_name in env.agents:
                    position = env.agents_data[agent_name]['position']
                    scaling_factor = env.agents_data[agent_name]['scaling_factor']
                    if position not in current_episode_agent_positions:
                        current_episode_agent_positions[position] = []
                    current_episode_agent_positions[position].append(f"{agent_name} scaling factor: {scaling_factor}")
                last_episode_agent_positions = current_episode_agent_positions

                break

        env.close()

        all_observed_states.extend(episode_observed_states)
        manager.collect_feedback()

        e += 1

    mean_deviation, std_deviation = calculate_deviation(all_observed_states, target_state)
    avg_bids_per_day = {i: sum(state[i] for state in all_observed_states) / len(all_observed_states) for i in range(7)}
    avg_position_per_scaling_factor = calculate_scaling_factor_positions(all_observed_states)
    operator_preferences = {day: {'min': target_state[day]['min'], 'max': target_state[day]['max']} for day in range(7)}

    results = {
        'avg_bids_per_day': avg_bids_per_day,
        'avg_position_per_scaling_factor': avg_position_per_scaling_factor,
        'operator_preferences': operator_preferences,
        'mean_deviation': mean_deviation,
        'std_deviation': std_deviation,
        'last_episode_agent_positions': last_episode_agent_positions
    }

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
