import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../csv/csv-2.csv')

x = data['Step']
y = data['Value']

alpha = 0.1

y_smoothed = y.ewm(alpha=alpha).mean()

percent_20 = 0.9 * y_smoothed

y_upper = y_smoothed + percent_20
y_lower = y_smoothed - percent_20

fig, ax = plt.subplots()
ax.fill_between(x, y_lower, y_upper, color='c', alpha=0.15)
ax.plot(x, y_smoothed, color='c', linewidth=2)

ax.set_xlim(-100, max(x)+100)
ax.set_ylim(0, max(y_upper))
ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
ax.set_xlabel('Steps', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Loss (Critic)', fontsize=14)

plt.savefig('loss_critic.svg', format='svg')
plt.show()
