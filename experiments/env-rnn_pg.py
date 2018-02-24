import argparse
import sys
import os
import os.path as op
import pathlib
import collections
import random
import string
import subprocess

import gym
from gym.error import Error

import numpy as np
import tensorflow as tf

import sirang.sirang as sirang

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# Ensure parent directory is on PATH
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import rnn_pg
from utils import empty_lists, load_yaml

Episode = collections.namedtuple("Episode", ['episode_length', 'loss', 'actions'])
# Defining experiment/model hyperparameter DB connector
experiment_saver = sirang.Sirang()


def plot_episode_lengths(episode_stats, env_name):
    plt.figure(figsize=(8, 6))
    plt.title('{env}: episode lengths'.format(env=env_name))

    episode_lengths = np.array([[e.episode_length for e in stats] for stats in episode_stats])
    trial_num = episode_lengths.shape[1]
    p_top = np.percentile(episode_lengths, q=75, axis=0)
    p_bottom = np.percentile(episode_lengths, q=25, axis=0)

    plt.plot(range(trial_num), np.median(episode_lengths, axis=0), color='orange', label='median episode length');
    if episode_lengths.shape[0] > 1:
        plt.fill_between(range(trial_num), p_top, p_bottom, alpha=0.3, color='orange');
    plt.legend()
    plt.show()


def median_episode_length(stats):
    return np.median([e.episode_length for e in stats])


def sliding_window_performance(scores, threshold, count_threshold, metric=np.mean):
    if len(scores) < count_threshold:
        return False
    else:
        scores.pop(0)
        return metric(scores) > threshold


@experiment_saver.dstore(
    db_name='env-training-runs', store=['graph', 'env', 'policy', 'verbose'], inversion=True, store_return=True)
def train(
    env, policy, graph=tf.get_default_graph(), n_episodes=700, gamma=0.78, n_updates=1,
    episode_target_length=np.inf, target_threshold_count=np.inf, load_checkpoint=False, 
    tensorflow_log_dir="./logs/log", tensorflow_checkpoint_path="./models/model.ckpt", 
    load_checkpoint_path=None, save_checkpoint=100, verbose=1, **kwargs):

    states, returns, actions, rewards, episode_stats, running_episode_length = empty_lists(6)

    with tf.Session(graph=graph) as sess:
        # Set up loggers and model checkpoint savers to paths specified above
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(tensorflow_log_dir, graph=sess.graph)
        # If working from previous savepoint load the saver; initialize variables.
        if load_checkpoint:
            saver.restore(sess, checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        
        # Start running episodes
        for k in range(n_episodes):
            state = env.reset()
            policy.initialize_rnn()
            done = False
            while not done:
                action_p = policy.action_p(state)
                states.append(state)
                action = policy.sample_action(action_p)
                state, reward, done, _ = env.step(action)
                
                rewards.append(reward)
                actions.append(action)
            
            # Apply episode termination cost
            rewards[-1] -= 30

            # Update rewards to yield step-wise returns
            while len(rewards) > 0:
                r = rewards.pop(0) 
                returns.append(r + sum(gamma**(i + 1) * r_t for i, r_t in enumerate(rewards)))

            # Update network
            for _ in range(n_updates):
                summary, loss = policy.update_policy(states, returns, actions)

            # Collect episode statistics
            running_episode_length.append(len(returns))
            episode_stats.append(Episode(len(returns), np.squeeze(loss), actions))
            print("Episode {} - Length : {}".format(k, len(returns)))
            actions, states, returns = empty_lists(3)

            # Check if early stopping condition is met
            terminate_condition = sliding_window_performance(
                running_episode_length, episode_target_length, target_threshold_count)

            # Save model if checkpoint reached or early stopping triggered
            if k % save_checkpoint == 0 or terminate_condition:
                save_path = saver.save(sess, tensorflow_checkpoint_path, global_step=k)
                writer.add_summary(summary, global_step=k)
                print("Session saved at: {}".format(save_path))
                if terminate_condition:
                    print("Training termination condition met; training aborted.")
                    break

            actions, states, returns = empty_lists(3)
       
    return {'median-episode-length': median_episode_length(episode_stats)}, episode_stats


if __name__ == "__main__":
    # Pass in configuration file and seed num (random starts).
    parser = argparse.ArgumentParser(description='Train PG-RNN on user specified task.')
    parser.add_argument('--config',
        help='Configuration file contains model and experiment hyperparameters')
    parser.add_argument('--seed-num', default=5, type=int,
        help='Number of different seeds to used for experiment.')
    parser.add_argument('--description', default="None supplied.", type=str,
        help='User supplied description to stored in MongoDB experiment document')

    # Load configuration file specifiying model and experiment hyperparameters as well
    # open-ai Environment to use.
    args = vars(parser.parse_args())
    # Note: argparsers sets unpassed flags to None.
    if args['config'] is not None:
        params = load_yaml(args['config'])
    else:
        raise Exception("config file parameter must be passed")

    # Set up sirang meta-data store
    if 'env' in params:
        params['env_name'] = params.pop('env')
        experiment_id = experiment_saver.store_meta(db_name="{env}-rnn_pg".format(
            env=params['env_name']), doc=args)
    else:
        raise Exception("'env' parameter not set in {config}".format(config=args['config']))
    # Make environment
    env = gym.make(params['env_name'])

    # Run training sessions for different seeds
    episode_stats = []
    for i in range(args['seed_num']):
        tf_graph = tf.Graph()
        with tf_graph.as_default():
            policy = rnn_pg.RNNPolicy(
                hidden_dim=params['hidden_dim'], env_state_size=len(env.observation_space.low), 
                action_space_dim=env.action_space.n, learning_rate=params['learning_rate'], 
                activation=tf.nn.relu, scope_name="model-{}".format(i))

            # Load up training run specific parameters to stored in MongoDB
            params.update(
                {'tensorflow_log_path': op.join(
                    params.get('tensorflow_log_dir', './logs'), "run-{}".format(i)), 
                 'tensorflow_checkpoint_path': op.join(
                    params.get('tensorflow_checkpoint_dir', './models'), "model-run-{}.ckpt".format(i)),
                 'scope_name': 'model-seed-{}'.format(i),
                 '_id': "{id}-{seed}".format(id=experiment_id, seed=i)})

            episode_stats.append(train(env=env, policy=policy, graph=tf_graph, **params))

    if params['generate_plot']:
        plot_episode_lengths(episode_stats, params['env_name'])
