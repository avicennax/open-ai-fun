import yaml
import sys

import numpy as np


def load_yaml(filename):
    with open(filename, 'r') as f_obj:
        return yaml.load(f_obj)


def empty_lists(n):
    return [[] for _ in range(n)]


def median_episode_length(stats):
    return np.median([e.episode_length for e in stats])


def sliding_window_performance(scores, threshold, count_threshold, metric=np.mean):
    if len(scores) < count_threshold:
        return False
    else:
        scores.pop(0)
        return metric(scores) > threshold