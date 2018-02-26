import argparse
import pathlib
import os.path as op
import re
import sys
import time

import numpy as np
import tensorflow as tf
import gym

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
import rnn_pg
import utils


def run(meta_graph_path, n_episodes=5, pause=2, **params):
    env = gym.make(params['env'])
    # Assumes that policy was generated via rnn_pg.py
    policy = rnn_pg.RNNPolicy(
        hidden_dim=params['hidden_dim'], env_state_size=len(env.observation_space.low), 
        action_space_dim=env.action_space.n, activation=tf.nn.relu, 
        scope_name=re.search("^.*(model-run-\d+)", path).group(1))

    with tf.Session() as sess:
        # Future option would be to load metagraph and pass it to the policy, this would
        # make sampling very easy, but training would be very difficult.
        importer = tf.train.Saver()
        # Assumes were TF version > 0.11.
        importer.restore(sess, op.splitext(meta_graph_path)[0])

        # Start running episodes
        env.render()
        for k in range(n_episodes):
            state = env.reset()
            policy.initialize_rnn(sess=sess)
            done = False
            while not done:
                time.sleep(0.1)
                action_p = policy.action_p(state, sess=sess)
                action = policy.sample_action(action_p)
                state, reward, done, _ = env.step(action)

            time.sleep(pause)


if __name__ == "__main__":
    # Pass in configuration file and seed num (random starts).
    parser = argparse.ArgumentParser(description='Run PG-RNN on user specified task.')
    parser.add_argument('--config',
        help='Configuration file contains model parameters')
    parser.add_argument('--meta-graph-path',
        help='Path for Tensorflow graph to be loaded up')
    parser.add_argument('--n-episodes', default=5, type=int,
        help='Number of different runs.')

    # Load configuration file specifiying model hyperparameters as well
    # open-ai Environment to use.
    args = vars(parser.parse_args())
    # Note: argparsers sets unpassed flags to None.
    if args['config'] is not None:
        params = utils.load_yaml(args['config'])
    else:
        raise Exception("config file parameter must be passed")

    if args['meta_graph_path'] is not None:
        meta_graph = args['meta_graph_path']
    else:
        raise Exception("meta-graph-path parameter must be passed")

    run(meta_graph_path=meta_graph, **params)