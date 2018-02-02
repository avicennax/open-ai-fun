import argparse
import sys
import collections
import random
import string
import subprocess

import gym
import numpy as np
import tensorflow as tf

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append('/Users/simonhaxby/projects/rl/openai-fun/src')
import rnn_pg
from utils import empty_lists, load_yaml

Episode = collections.namedtuple("Episode", ['episode_length', 'loss', 'actions'])


def plot_episode_lengths(episode_stats):
    plt.figure(figsize=(8, 6))
    plt.title('episode-length')
    for i, stats in enumerate(episode_stats):
        loss_t = [np.squeeze(e.loss) for e in stats]
        len_t = [e.episode_length for e in stats]
        plt.plot(range(len(len_t)), len_t, 'o', label='seed {}'.format(i));
    plt.legend()
    plt.show()


def train(
    env, policy, graph=tf.get_default_graph(), n_episodes=700, gamma=0.78, n_updates=1, load_checkpoint=False, 
    tensorflow_log_dir="/tmp/model/log", tensorflow_checkpoint_path="/tmp/model.ckpt", save_checkpoint=100, **kwargs):

    states, returns, actions, rewards, episode_stats = empty_lists(5)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(tensorflow_log_dir, graph=sess.graph)
        if load_checkpoint:
            saver.restore(sess, checkpoint_path)
        
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
            
            rewards[-1] -= 30

            while len(rewards) > 0:
                r = rewards.pop(0) 
                returns.append(r + sum(gamma**(i + 1) * r_t for i, r_t in enumerate(rewards)))
            
            for _ in range(n_updates):
                summary, loss = policy.update_policy(states, returns, actions)

            episode_stats.append(Episode(len(returns), np.squeeze(loss), actions))

            if k % save_checkpoint == 0:
                save_path = saver.save(sess, tensorflow_checkpoint_path, global_step=k)
                writer.add_summary(summary, global_step=k)
            
            print("Episode {} - Length : {}".format(k, len(returns)))
            actions, states, returns = empty_lists(3)
       
        print("Session saved at: {}".format(save_path))
        return episode_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PG-RNN on CartPole-v0 task.')
    parser.add_argument('--config', 
        help='Configuration file contains model and experiment hyperparameters')
    parser.add_argument('--seed-num', default=5, type=int,
        help='Number of different seeds to used for experiment.')

    args = vars(parser.parse_args())
    if args['config'] is not None:
        args.update(load_yaml(args['config']))

    env = gym.make('CartPole-v0')

    episode_stats = []
    for i in range(args['seed_num']):
        tf_graph = tf.Graph()
        with tf_graph.as_default():
            policy = rnn_pg.RNNPolicy(
                hidden_dim=args['hidden_dim'], env_state_size=4, action_space_dim=2,
                learning_rate=args['learning_rate'], activation=tf.nn.relu, scope_name="model-{}".format(i))
        args.update({'tensorflow_log_dir': "/tmp/model/log/seed-{}".format(i), 
                     'tensorflow_checkpoint_path': "/tmp/model/model-seed-{}.ckpt".format(i),
                     'scope_name': 'model-seed-{}'.format(i)})
        episode_stats.append(train(env=env, policy=policy, graph=tf_graph, **args))

    plot_episode_lengths(episode_stats)
