from flask import Flask, render_template, request, jsonify
import numpy as np
from scheduler import ResourceScheduler
import itertools
from service.yandex_explainability import yandex_explain
from service.gigachat_explainability import gigachat_explain
from rescheduling import (MAPPOAgent, MultiAgentSystemOperator, Client, format_logs_for_llm,
                          calculate_deviation, calculate_scaling_factor_positions)

app = Flask(__name__)


def filter_string(input_string):
    return input_string.replace('*', '').replace('#', '')


@app.route('/')
def index():
    return render_template('reschedule.html')


@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        data = request.json
        agent_id = int(data.get('agent_id'))

        clients = []
        for client_data in data['clients']:
            clients.append(Client(
                name=f'client_{len(clients)}',
                urgency=int(client_data['urgency']),
                completeness=int(client_data['completeness']),
                complexity=int(client_data['complexity'])
            ))

        target_state = {}
        for day, day_data in enumerate(data['targetState']):
            target_state[day] = {
                'min': int(day_data['min']),
                'max': int(day_data['max'])
            }

        initial_positions = {}
        for i in range(len(clients)):
            initial_positions[f'agent_{i}'] = int(data['initialPositions'][i])

        urgency_range = range(1, 4)
        completeness_range = range(0, 2)
        complexity_range = range(0, 2)
        options = list(itertools.product(urgency_range, completeness_range, complexity_range))
        selected_states = np.random.choice(len(options), size=12, replace=False)
        cargo_states = [options[i] for i in selected_states]

        agents = {f'agent_{i}': MAPPOAgent(obs_dim=11, action_dim=3) for i in range(len(cargo_states))}
        for i, agent in enumerate(agents):
            agents[agent].load_model(f'trained_model/actor_{i}.pth')

        manager = MultiAgentSystemOperator(list_of_clients=clients)
        manager.assign_agents({
            cargo_state: {'agent_name': agent_name, 'model_file': model_file}
            for cargo_state, (agent_name, model_file) in zip(cargo_states, agents.items())
        })

        all_observed_states = []
        last_episode_agent_positions = None
        e = 0

        while not all([client.satisfied for client in manager.clients]):
            env = ResourceScheduler(render_mode='terminal', max_agents=len(clients),
                                    max_days=7, max_episode_length=7)
            obs, _ = env.reset(options={
                'target_state': target_state,
                'agents_data': {
                    f'agent_{i}': {
                        'active': ~client.satisfied,
                        'base_reward': 1.0,
                        'window': 3,
                        'alpha': 2.0,
                        'urgency': client.urgency,
                        'completeness': client.completeness,
                        'complexity': client.complexity,
                        'mutation_rate': 0.0
                    } for i, client in enumerate(manager.clients)
                },
                'initial_positions': initial_positions
            })

            episode_observed_states = []
            step = 0

            while True:
                actions, step_logs = manager.get_actions(observations=obs, env=env, episode=e, step=step)
                next_obs, _, dones, truncations, next_info = env.step(actions)

                for log in step_logs:
                    manager._log_agent_step(
                        agent_id=log['agent_id'],
                        episode=e,
                        step=step,
                        action=log['action'],
                        info=log['pre_info'],
                        next_info=next_info[f'agent_{log["agent_id"]}'],
                        env=env
                    )

                step += 1
                obs = next_obs

                if any(dones.values()) or any(truncations.values()):
                    episode_observed_states.append(env.observed_state)

                    current_episode_agent_positions = {}
                    for agent_name in env.agents:
                        position = env.agents_data[agent_name]['position']
                        scaling_factor = env.agents_data[agent_name]['scaling_factor']
                        if position not in current_episode_agent_positions:
                            current_episode_agent_positions[position] = []
                        current_episode_agent_positions[position].append(
                            f"{agent_name} scaling factor: {scaling_factor}")
                    last_episode_agent_positions = current_episode_agent_positions
                    break

            all_observed_states.extend(episode_observed_states)
            manager.collect_feedback()
            e += 1

        mean_deviation, std_deviation = calculate_deviation(all_observed_states, target_state)
        avg_bids_per_day = {
            i: sum(state[i] for state in all_observed_states) / len(all_observed_states)
            for i in range(7)
        }

        avg_position_per_scaling_factor = calculate_scaling_factor_positions(all_observed_states)

        operator_preferences = {
            day: {'min': target_state[day]['min'], 'max': target_state[day]['max']}
            for day in range(7)
        }

        # Форматирование логов для LLM
        llm_logs = format_logs_for_llm(logs=manager.logs, env=env, agent_id=agent_id)

        # Получение объяснений от моделей

        yandex_explanation = filter_string(yandex_explain(logs=llm_logs, agent_id=agent_id))
        gigachat_explanation = filter_string(gigachat_explain(logs=llm_logs, agent_id=agent_id))

        result = {
            'stats': {
                'avgBidsPerDay': avg_bids_per_day,
                'avgPositionPerScalingFactor': avg_position_per_scaling_factor,
                'operatorPreferences': operator_preferences,
                'meanDeviation': mean_deviation * 100,
                'stdDeviation': std_deviation * 100,
                'lastEpisodeAgentPositions': last_episode_agent_positions
            },
            'explanations': {
                'yandex': yandex_explanation,
                'gigachat': gigachat_explanation
            }
        }

        return jsonify({'success': True, 'result': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=8000)
