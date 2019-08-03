import numpy as np
import tensorflow as tf

from . import utils


class RNNPolicy():
    def __init__(self, hidden_dim, env_state_size, action_space_dim, load_graph=False, learning_rate=0.01,
                 activation=tf.nn.elu, scope_name='policy-network', graph=None, **kwargs):

        if load_graph:
            self._load_graph()
        else:
            self._build_graph(hidden_dim, env_state_size,
                              action_space_dim, learning_rate, activation, scope_name)

        # sample action via passed p
        self.sample_action = lambda p: np.random.choice(
            range(action_space_dim), p=p)

    def _build_graph(self, hidden_dim, env_state_size, action_space_dim, learning_rate=0.01,
                     activation=tf.nn.elu, scope_name='policy-network', **kwargs):
        with tf.variable_scope(scope_name) as scope:
            # Size variables
            with tf.variable_scope('dimensions'):
                self.hidden_dim = hidden_dim
                self.env_state_dim = env_state_size
                self.action_space_dim = action_space_dim

            # model variables
            with tf.variable_scope('model-parameters'):
                self.rnn_cell = tf.contrib.rnn.BasicRNNCell(
                    hidden_dim, activation=activation)
                self.initial_state = tf.get_variable('rnn_init_state', [1, hidden_dim],
                                                     initializer=tf.contrib.layers.variance_scaling_initializer())
                self.output_weights = tf.get_variable('output_weights', [hidden_dim, action_space_dim],
                                                      initializer=tf.contrib.layers.variance_scaling_initializer())
                self.output_bias = tf.get_variable('output_bias', [action_space_dim],
                                                   initializer=tf.contrib.layers.variance_scaling_initializer())

            # single step
            self.env_state = tf.placeholder(
                tf.float32, [1, env_state_size], name="state")
            self.rnn_state = tf.placeholder(tf.float32, [1, hidden_dim])

            with tf.variable_scope('single-step-rnn'):
                self.rnn_state_val = None
                self.step_rnn, _ = self.rnn_cell(
                    self.env_state, self.rnn_state)
                self.action_probability = tf.nn.softmax(
                    tf.matmul(self.rnn_state, self.output_weights) + self.output_bias)

            # multiple episodes
            self.batch_size = tf.placeholder(tf.int32, name='max-episode-len')
            # returns ~ [n, max(epi_len)]
            self.returns = tf.placeholder(tf.float32, [None, None], 'returns')
            # env_states ~ [n, max(epi_len), env_state_size]
            self.env_states = tf.placeholder(
                tf.float32, [None, None, env_state_size], 'states')
            # actions ~ [n, max(epi_len), env_state_size]
            self.actions = tf.placeholder(tf.int32, [None, None, 3], 'actions')
            # tiling initial state
            self.initial_states = tf.tile(
                self.initial_state, multiples=[self.batch_size, 1])

            with tf.variable_scope('multi-step-rnn'):
                with tf.variable_scope('rnn'):
                    # rnn_states ~ [n, max(epi_len), hidden_dim]
                    self.rnn_states, _ = tf.nn.dynamic_rnn(
                        self.rnn_cell, inputs=self.env_states, initial_state=self.initial_states, dtype=tf.float32)

                with tf.variable_scope('action-p'):
                    # logits, action_probabilities ~ [n, max(epi_len), action_space_dim]
                    self.logits = tf.tensordot(self.rnn_states, self.output_weights, axes=[
                                               [2], [0]]) + self.output_bias
                    self.action_probabilities = tf.nn.softmax(self.logits)
                    # obs_action_probabilities ~ [n, max(epi_len)]
                    self.obs_action_probabilities = tf.gather_nd(
                        self.action_probabilities, self.actions)

            with tf.variable_scope('train'):
                # calculate path-wise likelihood ratios
                self.episodic_loss = tf.reduce_sum(
                    -tf.log(self.obs_action_probabilities + 1e-10) * self.returns, axis=1)
                # average over episodes
                self.loss = tf.reduce_mean(self.episodic_loss)
                self.optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=learning_rate)
                self.train_op = self.optimizer.minimize(
                    self.loss, global_step=tf.train.get_global_step())

            # summary variables
            with tf.variable_scope('summary'):
                tf.summary.tensor_summary('rnn-states', self.rnn_states)
                tf.summary.scalar('loss', self.loss)
                self.summary_op = tf.summary.merge_all()

    def _load_graph(self):
        pass

    def action_p(self, env_state, step_rnn=True, sess=None):
        sess = sess or tf.get_default_session()

        env_state = np.reshape(env_state, [1, self.env_state_dim])
        if step_rnn:
            feed_dict = {self.env_state: env_state,
                         self.rnn_state: self.rnn_state_val}
            self.rnn_state_val = sess.run(self.step_rnn, feed_dict)
        action_prob = sess.run(self.action_probability, {
                               self.rnn_state: self.rnn_state_val})
        return np.squeeze(action_prob)

    def update_policy(self, states, returns, actions, sess=None):
        """
        Parameters
        ----------
        states : [episodes [episode_len]]
        returns : [episodes [episode_len]]
        actions : [episodes [episode_len]]
        """
        sess = sess or tf.get_default_session()
        self.initialize_rnn()

        batch_size = len(states)
        max_episode_len = max([len(episode) for episode in states])
        states_with_zeros, returns_with_zeros, actions_with_zeros = utils.empty_lists(
            3)
        for episode_states, episode_returns, episode_actions in zip(states, returns, actions):
            states_with_zeros.append(episode_states +
                                     [np.zeros(self.env_state_dim) for _ in range(max_episode_len - len(episode_states))])
            actions_with_zeros.append(np.pad(
                episode_actions, mode='constant', pad_width=(0, max_episode_len - len(episode_actions))))
            returns_with_zeros.append(np.pad(
                episode_returns, mode='constant', pad_width=(0, max_episode_len - len(episode_returns))))

        states = np.array(states_with_zeros)
        returns = np.array(returns_with_zeros)
        actions = np.array([[[i, j, a] for j, a in enumerate(episode)]
                            for i, episode in enumerate(actions_with_zeros)])

        feed_dict = {self.env_states: states, self.returns: returns,
                     self.actions: actions, self.batch_size: batch_size}
        _, summary, loss = sess.run(
            [self.train_op, self.summary_op, self.loss], feed_dict)
        return summary, loss

    def initialize_rnn(self, sess=None):
        sess = sess or tf.get_default_session()
        self.rnn_state_val = sess.run(self.initial_state)
