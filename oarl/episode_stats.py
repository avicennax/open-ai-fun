import collections

Episode = collections.namedtuple(
    "Episode", ['episode_length', 'average_return', 'actions'])
