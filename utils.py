import yaml

def load_yaml(filename):
	with open(filename, 'r') as f_obj:
		return yaml.load(f_obj)


def empty_lists(n):
    return [[] for _ in range(n)]
