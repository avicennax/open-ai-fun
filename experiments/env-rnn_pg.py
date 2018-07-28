import argparse
import sys
import os
import os.path as op
import pathlib
import random
import string
import subprocess

import gym
from gym.error import Error

import numpy as np
import tensorflow as tf
from IPython import embed

import sirang

from oarl import (
    rnn_pg,
    rewards_funcs,
    viz,
    utils
)
from oarl.episode_stats import Episode

# Defining experiment/model hyperparameter DB connector
experiment_saver = sirang.Sirang()


@experiment_saver.dstore(
    db_name='env-training-runs', collection_name='rnn-pg',
    keep=['graph', 'env', 'policy', 'verbose'], inversion=True, store_return=True)
def train(
    env, policy, graph=tf.get_default_graph(), n_episodes=700, 
    gamma=0.78, n_updates=1,  episodes_per_update=5, 
    episode_target_length=np.inf, target_threshold_count=np.inf, 
    tensorflow_log_dir="./logs/log", 
    tensorflow_checkpoint_path="./models/model.ckpt", 
    load_checkpoint=False, load_checkpoint_path=None, save_checkpoint=100, 
    verbose=1, reward_func_params=None, reward_func='get_returns', **kwargs):

    all_states, all_actions, all_returns, episode_stats, running_episode_length = utils.empty_lists(5)
    if not reward_func_params:
        reward_func_params = {}

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
            states, actions, rewards = utils.empty_lists(3)
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

            # Get returns via user defined function.
            returns = getattr(rewards_funcs, reward_func)(
                **dict(reward_func_params, **{'rewards': rewards, 'gamma': gamma}))

            # Update state, action and returns collectors.
            all_states.append(states)
            all_actions.append(actions)
            all_returns.append(returns)

            # Collect episode statistics
            running_episode_length.append(len(returns))
            episode_stats.append(Episode(len(returns), np.mean(returns), actions))
            print("Episode {} - Length : {}".format(k, len(returns)))

            # Update network
            if k % episodes_per_update == 0 and k > 0:
                for _ in range(n_updates):
                    summary, loss = policy.update_policy(
                        all_states, all_returns, all_actions)
                all_states, all_returns, all_actions = utils.empty_lists(3)

            # Check if early stopping condition is met
            terminate_condition = utils.sliding_window_performance(
                running_episode_length, episode_target_length, 
                target_threshold_count)

            # Save model if checkpoint reached or early stopping triggered
            if (k % save_checkpoint == 0 and k > 0) or terminate_condition:
                save_path = saver.save(
                    sess, tensorflow_checkpoint_path, global_step=k)
                writer.add_summary(summary, global_step=k)
                print("Session saved at: {}".format(save_path))
                if terminate_condition:
                    print("Training termination condition met; training aborted.")
                    break
       
    median_episode_length = {
        'median-episode-length': utils.median_episode_length(episode_stats)}, 
    return median_episode_length, episode_stats


if __name__ == "__main__":
    # Pass in configuration file and seed num (random starts).
    parser = argparse.ArgumentParser(
        description='Train PG-RNN on user specified task.')
    parser.add_argument('--config',
        help='Configuration file contains model and experiment hyperparameters')
    parser.add_argument('--seed-num', default=5, type=int,
        help='Number of different seeds to used for experiment.')
    parser.add_argument('--description', default="None supplied.", type=str,
        help='User supplied description to stored in MongoDB experiment document')

    # Load configuration file specifiying model and experiment 
    #   hyperparameters as well open-ai Environment to use.
    args = vars(parser.parse_args())
    # Note: argparsers sets unpassed flags to None.
    if args['config'] is not None:
        params = utils.load_yaml(args['config'])
    else:
        raise Exception("config file parameter must be passed")

    # Set up sirang meta-data store
    if 'env' in params:
        params['env_name'] = params.pop('env')
        experiment_id = experiment_saver.store_meta(
            db_name='env-training-runs',
            collection_name="{env}-rnn_pg".format(env=params['env_name']), doc=args)
    else:
        raise Exception("'env' parameter not set in {config}".format(
            config=args['config']))
    # Make environment
    env = gym.make(params['env_name'])

    # Run training sessions for different seeds
    episode_stats = []
    for i in range(args['seed_num']):
        tf_graph = tf.Graph()
        with tf_graph.as_default():
            policy = rnn_pg.RNNPolicy(
                hidden_dim=params['hidden_dim'], env_state_size=len(
                    env.observation_space.low), 
                action_space_dim=env.action_space.n, 
                learning_rate=params['learning_rate'], 
                activation=tf.nn.relu, scope_name="model-run-{}".format(i))

            # Load up training run specific parameters to stored in MongoDB
            params.update(
                {'tensorflow_log_path': op.join(
                    params.get('tensorflow_log_dir', './logs'), "run-{}".format(i)), 
                 'tensorflow_checkpoint_path': op.join(
                    params.get(
                        'tensorflow_checkpoint_dir', './models'), 
                    "model-run-{}.ckpt".format(i)),
                 'scope_name': 'model-seed-{}'.format(i),
                 '_id': "{id}-{seed}".format(id=experiment_id, seed=i)})

            episode_stats.append(
                train(env=env, policy=policy, graph=tf_graph, **params))

    if params['generate_plot']:
        viz.plot_episode(episode_stats, params['env_name'], 'episode_length')

    # Open IPython session for interactive exploration
    embed()
