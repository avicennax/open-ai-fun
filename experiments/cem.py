# coding: utf-8
import argparse
import pathlib
import sys

import gym
import numpy as np

from oarl.utils import empty_lists
from oarl.rewards_funcs import get_returns


class CEMAgent(object):
    def __init__(self, thetas):
        self.thetas = thetas
    
    def action(self, state):
        x = np.dot(self.thetas, list(state) + [1])
        if x < 0:
            return 0
        else:
            return 1


def get_fittest(thetas, returns):
    top_r = np.percentile(returns, q=80)
    return thetas[returns >= top_r]


def train_agent(agent, env, n_episodes):
    states, rewards, actions, utilities = empty_lists(4)
    for k in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            states.append(state)
            action = agent.action(state)
            state, reward, done, _ = env.step(action)

            rewards.append(reward)
            actions.append(action)

        utilities.append(np.sum(get_returns(rewards, gamma)))
    return np.mean(utilities)


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
        params = utils.load_yaml(args['config'])
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
    state_size = env.observation_space.shape[0]
    mu = np.random.randn(state_size)
    thetas = np.random.randn(pop_size, state_size + 1) + mu
    utilities, pop_mean_util, mus = empty_lists(3)

    for k in range(iterations):
        mus.append(mu)
        for theta in thetas:
            utilities.append(train_agent(CEMAgent(theta), env, n_episodes))
        fittest = get_fittest(thetas, utilities)
        mu = np.mean(fittest, axis=0)
        thetas = np.random.randn(pop_size, state_size + 1) + mu
        pop_mean_util.append(np.mean(utilities))
        utilities = []
        print("iteration {k} complete -- mean utility: {u}".format(k=k, u=pop_mean_util[-1]))

