import matplotlib.pyplot as plt

# Исходные словари, представляющие конечные состояния эпизодов
final_states = [
    {0: 18, 1: 14, 2: 15, 3: 11, 4: 16, 5: 10, 6: 16},  # Episode 0
    {0: 24, 1: 17, 2: 16, 3: 14, 4: 9, 5: 11, 6: 9},   # Episode 1
    {0: 25, 1: 15, 2: 17, 3: 13, 4: 9, 5: 13, 6: 8},   # Episode 2
    {0: 24, 1: 16, 2: 16, 3: 13, 4: 9, 5: 12, 6: 10},  # Episode 3
    {0: 25, 1: 17, 2: 16, 3: 12, 4: 9, 5: 12, 6: 9},   # Episode 4
    {0: 25, 1: 16, 2: 16, 3: 12, 4: 9, 5: 13, 6: 9},   # Episode 5
    {0: 25, 1: 16, 2: 16, 3: 12, 4: 10, 5: 12, 6: 9},  # Episode 6
    {0: 26, 1: 16, 2: 16, 3: 12, 4: 9, 5: 12, 6: 9},   # Episode 7
    {0: 25, 1: 16, 2: 16, 3: 12, 4: 10, 5: 12, 6: 9},  # Episode 8
]


def transform_state(state):
    return {key: (value / 100) * 12 for key, value in state.items()}


transformed_states = [transform_state(state) for state in final_states]


def plot_histogram(data, episode_num):
    keys = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(10, 6))
    plt.bar(keys, values, color='skyblue')
    plt.xlabel('Days of the week')
    plt.ylabel('Number of slots (adjusted)')
    plt.title(f'Histogram of adjusted slots distribution for episode {episode_num+1}')
    plt.xticks(rotation=45)
    plt.ylim(0, 4)
    plt.tight_layout()
    plt.savefig(f'histogram_episode_{episode_num+1}.png')
    plt.show()


# Построение гистограмм для каждого эпизода на основе преобразованных данных
for i, state in enumerate(transformed_states):
    plot_histogram(state, i)
