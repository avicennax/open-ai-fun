#!/usr/bin/env python3
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
import os
import time
import yaml

import gym
import numpy as np
import tensorflow as tf

from oarl.cem import CMA_ES, CEMLogitAgent, get_fittest, train_agent
from oarl.utils import empty_lists, NestedDefaultDictionary

from metaflow import FlowSpec, IncludeFile, step


class CEMExperiment(FlowSpec):
    """
    Train logit on on user specified task via CEM
    """

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

        # Start timer
        self.start_time = time.time()

        self.next(self.run)

    @step
    def run(self):
        """Execute CMA Experiment"""

        # Master dictionary that tracks all run information with the following
        # key structure:
        #   'epoch' -> 'agent' ->
        #      {'parameters', 'run' -> [rewards, actions, states]}
        self.run_dict = NestedDefaultDictionary()

        # Write out logs for Tensorboard
        logdir = os.path.join(
            "logs",
            "cma",
            self.params["run"],
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            "metrics",
        )
        os.makedirs(logdir, exist_ok=True)

        file_writer = tf.summary.create_file_writer(logdir)
        file_writer.set_as_default()

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
        (
            utilities,
            self.pop_mean_util,
            self.pop_median_util,
            self.pop_var_util,
        ) = empty_lists(4)

        # Create metric iterables
        util_metrics = (
            self.pop_mean_util,
            self.pop_median_util,
            self.pop_var_util,
        )
        metric_names = ("Mean", "Median", "Variance")
        metric_funcs = (np.mean, np.median, np.var)

        # Define closure for static parameters
        def execute_training_runs(theta, epoch):
            agent = CEMLogitAgent(theta)
            return train_agent(
                agent, env, n_episodes, gamma, self._log_run(epoch, agent)
            )

        for k in range(iterations):
            tf.summary.scalar(
                f"CEM Population Covariance Trace",
                data=np.trace(pop_model.cov),
                step=k,
            )
            # For each agent run X many episodes - calculate mean utility
            # across the X runs for each agent.
            utilities = [execute_training_runs(theta, k) for theta in thetas]

            # Keep "best" agents in population sample and refit the
            # population generator using utility weighted parameters of
            # agents in sample.
            fittest, _ = get_fittest(thetas, np.array(utilities))
            pop_model.update_generator(fittest, _)

            # Record population returns metrics
            for collection, metric_name, func in zip(
                util_metrics, metric_names, metric_funcs
            ):
                collection.append(func(utilities))
                tf.summary.scalar(
                    f"Population Returns {metric_name}",
                    data=collection[-1],
                    step=k,
                )

            print(
                "Iteration {k} complete -- mean utility: {u}".format(
                    k=k, u=self.pop_mean_util[-1]
                ),
                flush=True,
            )
            # Sample new population from new population parameter generator
            thetas = pop_model.generate_population(pop_size)
            utilities = []

        self.next(self.end)

    @step
    def end(self):
        # Log experiment run time
        self.end_time = time.time()
        run_td = timedelta(seconds=self.start_time - self.end_time)
        self.run_time = str(run_td)

    def _log_run(self, epoch, agent):
        def callback(rewards, actions, states):
            # TODO: determine a better strategy for referencing
            # the agents that can work in a distributed environment.
            if not self.run_dict[epoch][hash(agent)]:
                self.run_dict[epoch][hash(agent)] = {
                    "parameters": agent.thetas,
                    "runs": [
                        {
                            "rewards": rewards,
                            "actions": actions,
                            "states": states,
                        }
                    ],
                }
            else:
                self.run_dict[epoch][hash(agent)]["runs"].append(
                    {"rewards": rewards, "actions": actions, "states": states}
                )

        return callback


if __name__ == "__main__":
    CEMExperiment()
