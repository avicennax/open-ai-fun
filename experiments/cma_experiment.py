#!/usr/bin/env python
import yaml

import numpy as np

import gym

from oarl.cem import CMA_ES, CEMLogitAgent, get_fittest, train_agent
from oarl.utils import empty_lists

from metaflow import FlowSpec, IncludeFile, step


class CEMExperiment(FlowSpec):
    """Train logit on on user specified task via CEM"""

    config = IncludeFile(
        "config",
        required=True,
        help="Configuration file contains model and experiment hyperparameters",
    )

    @step
    def start(self):
        """Initialize experiment parameters"""

        # Load experiment configuration parameters into memory
        self.params = yaml.safe_load(self.config)

        # TODO: Fix Tensorflow writer
        # Define Tensorflow writer
        # self.writer = tf.summary.FileWriter(".")

        self.next(self.run)

    @step
    def run(self):
        """Execute CMA Experiment"""
        
        # Experiment/ES parameters
        gamma = self.params["gamma"]
        n_episodes = self.params["n-episodes"]
        pop_size = self.params["pop-size"]
        iterations = self.params["iterations"]
        env = gym.make(self.params["env"])

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
            # TODO: Fix TF reporting
            # tf.summary.scalar("Population Mean Returns", pop_mean_util[-1], k)
            print(
                "iteration {k} complete -- mean utility: {u}".format(
                    k=k, u=pop_mean_util[-1]
                )
            )

            # Sample new population from new population parameter generator
            thetas = pop_model.generate_population(pop_size)

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    CEMExperiment()
