import yaml
import sys


def load_yaml(filename):
    with open(filename, 'r') as f_obj:
        return yaml.load(f_obj)


def empty_lists(n):
    return [[] for _ in range(n)]


# def verbose_print(verbose, out):
#     if verbose == 1:
#         print(out)
#     elif verbose == 2:
#         print(out, file=sys.stderr)
