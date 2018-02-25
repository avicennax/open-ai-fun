import numpy as np
import tensorflow as tf

class RNNPolicy():
    def __init__(self, hidden_dim, env_state_size, action_space_dim, load_graph=False, learning_rate=0.01, 
        activation=tf.nn.elu, scope_name='policy-network', graph=None, **kwargs):
    
        if load_graph:
            self._load_graph()
        else:
            self._build_graph(hidden_dim, env_state_size, action_space_dim, learning_rate, activation, scope_name)

        # sample action via passed p
        self.sample_action = lambda p: np.random.choice(range(action_space_dim), p=p)

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
                self.rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_dim, activation=activation)
                self.initial_state = tf.get_variable('rnn_init_state', [1, hidden_dim], 
                    initializer=tf.contrib.layers.variance_scaling_initializer())
                self.output_weights = tf.get_variable('output_weights', [hidden_dim, action_space_dim],
                    initializer=tf.contrib.layers.variance_scaling_initializer())
                self.output_bias = tf.get_variable('output_bias', [action_space_dim],
                    initializer=tf.contrib.layers.variance_scaling_initializer())

            # single step
            self.env_state = tf.placeholder(tf.float32, [1, env_state_size], name="state")
            self.rnn_state = tf.placeholder(tf.float32, [1, hidden_dim])

            with tf.variable_scope('single-step-rnn'):
                self.rnn_state_val = None
                self.step_rnn, _ = self.rnn_cell(self.env_state, self.rnn_state)
        
                self.action_probability = tf.nn.softmax(tf.matmul(self.rnn_state, self.output_weights) + self.output_bias)

            # multiple states
            self.actions = tf.placeholder(tf.int32, [None, 2])
            self.returns = tf.placeholder(tf.float32, [None], 'episode_returns')
            self.env_states = tf.placeholder(tf.float32, [1, None, env_state_size], 'episode_states')
            
            with tf.variable_scope('multi-step-rnn'):
                with tf.variable_scope('rnn'):
                    self.rnn_states, _ = tf.nn.dynamic_rnn(
                        self.rnn_cell, inputs=self.env_states, initial_state=self.initial_state, dtype=tf.float32)
                    self.rnn_states_reshaped = tf.squeeze(self.rnn_states) # [?, hidden_dim]

                with tf.variable_scope('action-p'):
                    self.logits = tf.matmul(self.rnn_states_reshaped, self.output_weights) + self.output_bias # [?, action_space_dim]
                    self.action_probabilities = tf.nn.softmax(self.logits)
                    self.obs_action_probabilities = tf.gather_nd(self.action_probabilities, self.actions) # [?]

            with tf.variable_scope('train'):
                self.loss = tf.tensordot(-tf.log(self.obs_action_probabilities), self.returns, 1)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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
            feed_dict = {self.env_state: env_state, self.rnn_state: self.rnn_state_val}
            self.rnn_state_val = sess.run(self.step_rnn, feed_dict)
        action_prob = sess.run(self.action_probability, {self.rnn_state: self.rnn_state_val})
        return np.squeeze(action_prob)
    
    def update_policy(self, states, returns, actions, sess=None):
        sess = sess or tf.get_default_session()
        self.initialize_rnn()
        
        states = np.reshape(np.array(states), [1, len(states), self.env_state_dim])
        returns = np.array(returns)
        actions = np.array([[i, a] for i, a in enumerate(actions)])
        
        feed_dict = {self.env_states: states, self.returns: returns, self.actions: actions}
        _, summary, loss = sess.run([self.train_op, self.summary_op, self.loss], feed_dict)
        return summary, loss
        
    def initialize_rnn(self, sess=None):
        sess = sess or tf.get_default_session()
        self.rnn_state_val = sess.run(self.initial_state)
