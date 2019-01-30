import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import itertools
from random import shuffle
# from tensorflow_probability import distributions as tfd
tfd = tfp.distributions

class CategoricalMixture(object):
    
    def __init__(self, K, L, m):
        self.K = K  # number of components
        self.L = L  # sequence length
        self.m = m  # number of categories
        
        self.z_logits = tf.Variable(tf.random.uniform((K,)), dtype=tf.float32)
        self.p_logits = [tf.Variable(tf.random.uniform((L, m,)), dtype=tf.float32) for _ in range(K)]
        self.z = tf.nn.softmax(self.z_logits)
        self.ps = [tf.nn.softmax(self.p_logits[i]) for i in range(K)]
        self.components = [tfd.Independent(tfd.Categorical(probs=self.ps[i]), reinterpreted_batch_ndims=1) for i in range(K)]
        self.model = tfd.Mixture(
            cat = tfd.Categorical(probs=self.z),
            components=self.components           
        )
        
        self.X_train = tf.placeholder(name="X_train",shape=[None, L], dtype=tf.float32)
        self.weights = tf.placeholder(name="loss_weights", shape=[None], dtype=tf.float32)
        self.loss = -tf.reduce_mean(self.weights * self.model.log_prob(self.X_train))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def _iterate_minibatches(self, inputs1, inputs2, batch_size, shuffle=True):
        if shuffle:
            indices = np.arange(inputs1.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs1.shape[0] - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs1[excerpt], inputs2[excerpt]
        
    def train(self, X, W, epochs=100, batch_size=10, shuffle=True, verbose=False, one_hot=False, print_every=100):
        if one_hot:
            X = np.argmax(X, axis=-1)
        for t in range(epochs):
            e_loss = 0
            n_batches = 0
            for batch in self._iterate_minibatches(X, W, batch_size, shuffle=shuffle):
                xi, wi = batch
                _, np_loss = self.sess.run([self.train_op, self.loss], feed_dict={self.X_train: xi, self.weights:wi})
                e_loss += np_loss
                n_batches += 1
            if verbose:
                if t % print_every == 0 or t == epochs-1:
                    print("Training loss at %i/%i: %.3f" % (t, epochs, e_loss/n_batches))
                
    def sample(self, n, one_hot=False):
        samples = self.model.sample(n).eval(session=self.sess)
        if one_hot:
            samples_one_hot = np.zeros((n, self.L, self.m))
            samples_one_hot[np.arange(n).reshape(n, 1), np.arange(self.L), samples] = 1
            samples = samples_one_hot
        return samples
    
    def log_prob(self, x, one_hot=False):
        if one_hot:
            x_eval = np.argmax(x, axis=-1)
        else:
            x_eval = x
        return self.model.log_prob(x_eval).eval(session=self.sess)
    