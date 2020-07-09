import numpy as np
from keras import backend as K

from seqtools import SequenceTools

"""
This module contains a number of custom Keras loss functions
"""


def zero_loss(y_true, y_pred):
    """Returns zero"""
    return K.zeros_like(y_true)

def identity_loss(y_true, y_pred):
    """Returns the predictions"""
    return y_pred

def summed_binary_crossentropy(y_true, y_pred):
    """ Negative log likehood of binomial distribution """
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)  # default is mean over last axis


def summed_categorical_crossentropy(y_true, y_pred):
    """ Negative log likelihood of categorical distribution """
    return K.sum(K.categorical_crossentropy(y_true, y_pred), axis=-1)


def get_gaussian_nll(variance=1.):
    """  Returns gaussian negative log likelihood loss function """

    def gaussian_nll(y_true, y_pred):
        return K.sum(0.5 * K.log(2 * np.pi) + 0.5 * K.log(variance) + (0.5 / variance) * K.square(y_true - y_pred),
                     axis=-1)

    return gaussian_nll

def neg_log_likelihood(y_true, y_pred):
    """Returns negative log likelihood of Gaussian"""
    y_true = y_true[:, 0]
    mean = y_pred[:, 0]
    variance = K.softplus(y_pred[:, 1]) + 1e-6
    log_variance = K.log(variance)
    return 0.5 * K.mean(log_variance, axis = -1) + 0.5 * K.mean(K.square(y_true - mean) / variance, axis = -1) + 0.5 * K.log(2 * np.pi)


def get_uncertainty_loss(variance=1.):
    """Resturns gaussian loss """
    def uncertainty_loss(v_true, v_pred):
        return K.sum((0.5 / variance) * K.square(v_true - v_pred), axis=-1)

    return uncertainty_loss


def get_gaussian_nll_for_log_pred(variance=1.):
    """Returns gaussian negative log likelihood loss function"""

    def gaussian_log_nll(y_true, y_pred):
        return K.sum(
            0.5 * K.log(2 * np.pi) + 0.5 * K.log(variance) + (0.5 / variance) * K.square(y_true - K.exp(y_pred)),
            axis=-1)

    return gaussian_log_nll


