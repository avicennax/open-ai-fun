# coding: utf-8
import argparse
import pathlib
import sys

import gym
import numpy as np

from oarl.cem import (
    CEM_Mean, 
    CMA_ES,
    CEMLogitAgent, 
    get_fittest,
    train_agent
)
from oarl.utils import empty_lists, load_yaml
from oarl.rewards_funcs import get_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train logit on on user specified task via CEM.')
    parser.add_argument('--config',
        help='Configuration file contains model and experiment hyperparameters')
    args = vars(parser.parse_args())

    # Load configuration file specifiying experiment hyperparameters as well
    # open-ai Environment to use.
    args = vars(parser.parse_args())
    # Note: argparsers sets unpassed flags to None.
    if args['config'] is not None:
        params = load_yaml(args['config'])
    else:
        raise Exception("config file parameter must be passed")

    # Experiment/ES parameters
    gamma = params['gamma']
    n_episodes = params['n-episodes']
    pop_size = params['pop-size']
    iterations = params['iterations']
    env = gym.make(params['env'])

    cont = None

    # Initializing experiment
    param_size = env.observation_space.shape[0] + 1
    pop_model = CMA_ES(param_size)
    thetas = pop_model.generate_population(pop_size)
    utilities, pop_mean_util, mus = empty_lists(3)

    for k in range(iterations):
        mus.append(pop_model.mu)
        for theta in thetas:
            utilities.append(
                train_agent(CEMLogitAgent(theta), env, n_episodes, gamma)
            )
        
        # Fit a Gaussian parameter generator with unit variance
        fittest, _ = get_fittest(thetas, np.array(utilities))
        pop_model.update_generator(fittest, _)
        
        # Record population fittest
        pop_mean_util.append(np.mean(utilities))
        utilities = []
        print("iteration {k} complete -- mean utility: {u}".format(
            k=k, u=pop_mean_util[-1]))

        # Sample new population from new population parameter generator
        thetas = pop_model.generate_population(pop_size)


