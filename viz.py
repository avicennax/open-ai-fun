import re
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def plot_episode(episode_stats, env_name, ylabel):
    plt.figure(figsize=(8, 6))
    plt.title('{env}: {label}'.format(env=env_name, label=re.sub('[_-]', ' ', ylabel))

    episode_lengths = np.array([[getattr(e, ylabel) for e in stats] for stats in episode_stats])
    trial_num = episode_lengths.shape[1]
    p_top = np.percentile(episode_lengths, q=75, axis=0)
    p_bottom = np.percentile(episode_lengths, q=25, axis=0)

    plt.plot(range(trial_num), np.median(episode_lengths, axis=0), 
        color='orange', label='median episode length');
    if episode_lengths.shape[0] > 1:
        plt.fill_between(range(trial_num), p_top, p_bottom, alpha=0.3, color='orange');
    plt.xlabel('episode number')
    plt.legend()
    plt.show()
