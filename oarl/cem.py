# coding: utf-8

from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np

from .utils import empty_lists
from .rewards_funcs import get_returns

#  .___  ___.   ______    _______   _______  __          _______.
#  |   \/   |  /  __  \  |       \ |   ____||  |        /       |
#  |  \  /  | |  |  |  | |  .--.  ||  |__   |  |       |   (----`
#  |  |\/|  | |  |  |  | |  |  |  ||   __|  |  |        \   \
#  |  |  |  | |  `--'  | |  '--'  ||  |____ |  `----.----)   |
#  |__|  |__|  \______/  |_______/ |_______||_______|_______/


class CEM_Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, param_shape):
        pass

    @abstractmethod
    def update_generator(self, population, utilities):
        pass

    @abstractmethod
    def generate_population(self, pop_size):
        pass


class CEM_Mean(CEM_Model):
    """
    Vanilla CEM using Gaussian population generator with
    identity covariance.
    """

    def __init__(self, param_shape):
        self.mu = np.random.randn(param_shape)
        self.param_shape = param_shape

    def update_generator(self, population, utilities):
        self.mu = np.mean(population, axis=0)

    def generate_population(self, pop_size):
        return np.random.randn(pop_size, self.param_shape) + self.mu


class CEM_MeanWeighted(CEM_Model):
    """
    Variant of CEM using weighted samples; with being calculated
    using the estimated expected returns of the population
    survivors.
    """

    def __init__(self, param_shape, lamb=1e-3):
        self.mu = np.random.randn(param_shape)
        self.param_shape = param_shape
        self.lamb = lamb

    def update_generator(self, population, utilities):
        weights = np.exp(self.lamb * utilities)
        weighted_pop = (population.T * weights).T / np.mean(weights)
        self.mu = np.mean(weighted_pop, axis=0)

    def generate_population(self, pop_size):
        return np.random.randn(pop_size, self.param_shape) + self.mu


class CEM_MeanCov(CEM_Model):
    """
    Variant of CEM using updates to covariance matrix.
    """

    def __init__(self, param_shape):
        self.mu = np.random.randn(param_shape)
        self.cov = np.random.exponential(size=(param_shape, param_shape))

    def update_generator(self, population, utilities):
        self.mu = np.mean(population, axis=0)
        # Correct for bias with (N-1) denom via ddof arg.
        self.cov = np.cov(population.T, ddof=1)

    def generate_population(self, pop_size):
        return np.random.multivariate_normal(self.mu, self.cov, pop_size)


class CMA_ES(CEM_Model):
    """
    CMA-ES; Covariance Matrix Adapation Evolution Strategy:
        A variant on CEM thats updates the population generator's
        covariance matrix and using weighted samples, specifically
        using their estimated expected returns.
    """

    def __init__(self, param_shape, lamb=1e-3):
        self.mu = np.random.randn(param_shape)
        self.cov = np.random.exponential(size=(param_shape, param_shape))
        self.lamb = lamb

    def update_generator(self, population, utilities):
        weights = np.exp(self.lamb * utilities)
        weighted_pop = (population.T * weights).T / np.mean(weights)
        self.mu = np.mean(weighted_pop, axis=0)
        # Correct for bias with (N-1) denom via ddof arg.
        self.cov = np.cov(weighted_pop.T, ddof=1)

    def generate_population(self, pop_size):
        return np.random.multivariate_normal(self.mu, self.cov, pop_size)


#      ___       _______  _______ .__   __. .___________.    _______.
#     /   \     /  _____||   ____||  \ |  | |           |   /       |
#    /  ^  \   |  |  __  |  |__   |   \|  | `---|  |----`  |   (----`
#   /  /_\  \  |  | |_ | |   __|  |  . `  |     |  |        \   \
#  /  _____  \ |  |__| | |  |____ |  |\   |     |  |    .----)   |
# /__/     \__\ \______| |_______||__| \__|     |__|    |_______/


class CEMAgent(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, thetas):
        pass

    @abstractmethod
    def action(self, state):
        pass


class CEMLogitAgent(CEMAgent):
    def __init__(self, thetas):
        self.thetas = thetas

    def action(self, state):
        x = np.dot(self.thetas, list(state) + [1])
        if x < 0:
            return 0
        else:
            return 1


def train_agent(agent, env, n_episodes, gamma, episode_callback=None):
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

        # Call callback on episode metrics
        if episode_callback:
            episode_callback(
                *[deepcopy(it) for it in (rewards, actions, states)]
            )

        utilities.append(np.sum(get_returns(rewards, gamma)))
        rewards, actions, states = empty_lists(3)

    return np.mean(utilities)


def get_fittest(thetas, returns, q=80):
    top_r = np.percentile(returns, q=q)
    return thetas[returns >= top_r], returns[returns >= top_r]
