from __future__ import print_function

import functools
import tensorflow as tf
import numpy as np

def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def weight_variable(shape, name="W") :
  var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
  return var

def bias_variable(shape, name="b") :
  initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
  return tf.Variable(initial, name=name)

def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=64, 
              activation=tf.tanh,
              output_activation=None):

    with tf.variable_scope(scope):
        x = input_placeholder
        for i in range(1,n_layers+1):
            x = tf.layers.dense(x, units=size, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="layer_"+str(i))

        output = tf.layers.dense(x, units=output_size, activation=output_activation, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="output")
    return output

class BehaviourCloning:

    def __init__(self, obs, a, n_layers=1, hidden_size=64):       # obs -> (None, obs_dim) | ats -> (None, ats_dim)
        self.obs = obs
        self.a = a

        self.obs_dim = obs.get_shape()[1]
        self.a_dim = a.get_shape()[1]

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.policy
        self.loss
        self.optimize


    @define_scope
    def policy(self):
        return build_mlp(self.obs, self.a_dim, "Policy_Network", n_layers=self.n_layers, size=self.hidden_size, activation=tf.tanh)

    @define_scope
    def loss(self):
        return tf.losses.mean_squared_error(
                  self.a,
                  self.policy)

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(0.01)
        return optimizer.minimize(self.loss)



def train(env, expert_data, batch_size=50, num_itrs=500, log_itr=100, num_layers=1, hidden_size=64):
    
    trn_size = len(expert_data['observations'])

    obs_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    obs = tf.placeholder(tf.float32, [None, obs_dim])
    a   = tf.placeholder(tf.float32, [None, a_dim])

    model = BehaviourCloning(obs, a, num_layers, hidden_size)

    saver = tf.train.Saver()

    with tf.Session() as sess :
        writer = tf.summary.FileWriter('log', sess.graph)

        tf.summary.scalar('Loss', model.loss)
        merged = tf.summary.merge_all()


        sess.run(tf.global_variables_initializer())
        
        itr = 0
        while itr < num_itrs :
          itr += 1
          indices = np.random.choice(trn_size, batch_size)
          obs_sample = np.array(map(lambda i : expert_data['observations'][i], indices))
          a_sample   = np.array(map(lambda i : expert_data['actions'][i], indices))

          a_sample = a_sample.reshape([batch_size, a_dim])

          _, summary = sess.run([model.optimize, merged], {obs:obs_sample, a:a_sample})

          writer.add_summary(summary, itr)


          if not itr % log_itr :
            print ("itr[%d] Loss - %.3f" % (itr, sess.run(model.loss, {obs:obs_sample, a:a_sample})))

        save_path = saver.save(sess, "model/model.ckpt")

        print ("Model saved in : ", save_path)

        writer.close()

    return obs, a, model
