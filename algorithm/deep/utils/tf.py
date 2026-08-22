import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from ..exceptions import WrongVarianceCovarianceMatrixException
from keras.optimizers import Optimizer as KerasOptimizer
from nevergrad.optimization import Optimizer as NevergradOptimizer



from .common import (
    set_parameters__ranges,
    set_parameters__initials,
    fill_matrix,
    spiral_flat_from_progressive,
    iterate_over_elements_below_diagonal,
    is_valid_matrix,
    is_positive_definite,
    is_symmetric,
    normalize_value,
    denormalize_value,
    normalize_value_log,
    denormalize_value_log,
)

__is_valid_matrix = is_valid_matrix
__is_positive_definite = is_positive_definite
__is_symmetric = is_symmetric


def scaleAndFlat_matrix(flat_m, dtype=np.float32):
    size = int(0.5*(math.sqrt(1+8*flat_m.shape[0])-1))
    m = np.zeros((size,size), dtype=dtype)
    i_f = 0
    for i_c in range(size):
        for i_r in range(i_c+1):
            m[i_c,i_r] = flat_m[i_f]
            i_f += 1
    for i_r in range(size):
        for i_c in range(i_r,size):
            m[i_r,i_c] = m[i_c,i_r]
            i_f += 1
    if not is_valid_matrix(m):
        raise WrongVarianceCovarianceMatrixException(m)
    chol_factorization = tf.linalg.cholesky(m)
    flat_chol_factorized_matrix = tfp.math.fill_triangular_inverse(chol_factorization)
    return flat_chol_factorized_matrix.numpy()


class Triangular(tf.Module):
    def __init__(self, low, mode, high):
        super(Triangular, self).__init__()
        self.low = low
        self.mode = mode
        self.high = high
        self.uniform1 = tfp.distributions.Uniform(self.low, self.mode)
        self.uniform2 = tfp.distributions.Uniform(self.mode, self.high)

    def sample(self, sample_shape=()):
        p = tf.random.uniform(sample_shape)
        return tf.where(p < (self.mode - self.low) / (self.high - self.low),
                        self.uniform1.sample(sample_shape),
                        self.uniform2.sample(sample_shape))


class MinMaxNorm_ElementWise(keras.constraints.Constraint):
    def __init__(self, min_values, max_values):
        self.min_values = min_values
        self.max_values = max_values

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_values, self.max_values)

    def get_config(self):
        return {'min_values': self.min_values, 'max_values': self.max_values}


def is_keras_optimizer(opt):
    return isinstance(opt, KerasOptimizer)

def is_nevergrad_optimizer(opt):
    return isinstance(opt, NevergradOptimizer)
