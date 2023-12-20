import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from .exceptions import WrongVarianceCovarianceMatrixException
from keras.optimizers import Optimizer as KerasOptimizer
from nevergrad.optimization import Optimizer as NevergradOptimizer



def set_parameters__ranges(mode, param):
    if mode == "multivariatenormal":
        if param == "mean vector":
            return [
                # element: 0
                [-1, 1],
                # element: 1
                [float('-inf'), float('inf')],
                # element: 2
                [float('-inf'), float('inf')],
                # element: 3
                [float('-inf'), float('inf')],
                # element: 4
                [-1, 1],
                # element: 5
                [0, 100],
                # element: 6
                [0, float('inf')]]
        if param == "variancecovariance_matrix":
            return [
                # element: 0,0
                [0, 0.4],
                # element: 1,0
                [-100, 100],
                # element: 1,1
                [0, 10],
                # element: 2,0
                [-100, 100],
                # element: 2,1
                [-100, 100],
                # element: 2,2
                [0, 10],
                # element: 3,0
                [-100, 100],
                # element: 3,1
                [-100, 100],
                # element: 3,2
                [-100, 100],
                # element: 3,3
                [0, 10],
                # element: 4,0
                [-100, 100],
                # element: 4,1
                [-100, 100],
                # element: 4,2
                [-100, 100],
                # element: 4,3
                [-100, 100],
                # element: 4,4
                [0, 0.4],
                # element: 5,0
                [-100, 100],
                # element: 5,1
                [-100, 100],
                # element: 5,2
                [-100, 100],
                # element: 5,3
                [-100, 100],
                # element: 5,4
                [-100, 100],
                # element: 5,5
                [0, 25],
                # element: 6,0
                [-100, 100],
                # element: 6,1
                [-100, 100],
                # element: 6,2
                [-100, 100],
                # element: 6,3
                [-100, 100],
                # element: 6,4
                [-100, 100],
                # element: 6,5
                [-100, 100],
                # element: 6,6
                [0, 10]
            ]
    if mode == "uniform":
        if param == "lowers":
            return [
                # element: 0
                [-1, 1],
                # element: 1
                [float('-inf'), float('inf')],
                # element: 2
                [float('-inf'), float('inf')],
                # element: 3
                [float('-inf'), float('inf')],
                # element: 4
                [-1, 1],
                # element: 5
                [0, 100],
                # element: 6
                [0, float('inf')]]
        if param == "uppers":
            return [
                # element: 0
                [-1, 1],
                # element: 1
                [float('-inf'), float('inf')],
                # element: 2
                [float('-inf'), float('inf')],
                # element: 3
                [float('-inf'), float('inf')],
                # element: 4
                [-1, 1],
                # element: 5
                [0, 100],
                # element: 6
                [0, float('inf')]]
    if mode == "triangular":
        if param == "lowers":
            return [
                # element: 0
                [-1, 1],
                # element: 1
                [float('-inf'), float('inf')],
                # element: 2
                [float('-inf'), float('inf')],
                # element: 3
                [float('-inf'), float('inf')],
                # element: 4
                [-1, 1],
                # element: 5
                [0, 100],
                # element: 6
                [0, float('inf')]]
        if param == "modes":
            return [
                # element: 0
                [-1, 1],
                # element: 1
                [float('-inf'), float('inf')],
                # element: 2
                [float('-inf'), float('inf')],
                # element: 3
                [float('-inf'), float('inf')],
                # element: 4
                [-1, 1],
                # element: 5
                [0, 100],
                # element: 6
                [0, float('inf')]]
        if param == "uppers":
            return [
                # element: 0
                [-1, 1],
                # element: 1
                [float('-inf'), float('inf')],
                # element: 2
                [float('-inf'), float('inf')],
                # element: 3
                [float('-inf'), float('inf')],
                # element: 4
                [-1, 1],
                # element: 5
                [0, 100],
                # element: 6
                [0, float('inf')]]
    if mode == "univariatenormal":
        if param == "means":
            return [
                # element: 0
                [-1, 1],
                # element: 1
                [float('-inf'), float('inf')],
                # element: 2
                [float('-inf'), float('inf')],
                # element: 3
                [float('-inf'), float('inf')],
                # element: 4
                [-1, 1],
                # element: 5
                [0, 100],
                # element: 6
                [0, float('inf')]]
        if param == "variances":
            return [
                # element: 0
                [0, 0.4],
                # element: 1
                [0, 10],
                # element: 2
                [0, 10],
                # element: 3
                [0, 10],
                # element: 4
                [0, 0.4],
                # element: 5
                [0, 25],
                # element: 6
                [0, 10]]


def set_parameters__initials(mode, param, ranges):
# possible to implement other strategies.
    if mode == "multivariatenormal":
        if param == "mean vector":
            return [(r[1]+r[0])/2 for r in ranges]
        if param == "variancecovariance_matrix":
            return [(r[1]+r[0])/2 for r in ranges]
    if mode == "uniform":
        if param == "lowers":
            return [(r[1]+r[0])/2 for r in ranges]
        if param == "uppers":
            return [(r[1]+r[0])/2 for r in ranges]
    if mode == "triangular":
        if param == "lowers":
            return [(r[1]+r[0])/2 for r in ranges]
        if param == "modes":
            return [(r[1]+r[0])/2 for r in ranges]
        if param == "uppers":
            return [(r[1]+r[0])/2 for r in ranges]
    if mode == "univariatenormal":
        if param == "means":
            return [(r[1]+r[0])/2 for r in ranges]
        if param == "variances":
            return [(r[1]+r[0])/2 for r in ranges]


def fill_matrix(flat_ms):
    ms = []
    for i in range(flat_ms.shape[0]):
        flat_m = flat_ms[i]
        size = int(0.5*(math.sqrt(1+8*flat_m.shape[0])-1))
        m = np.zeros((size,size))
        i_f = 0
        for i_c in range(size):
            for i_r in range(i_c+1):
                m[i_c,i_r] = flat_m[i_f]
                i_f += 1
        for i_r in range(size):
            for i_c in range(i_r,size):
                m[i_r,i_c] = m[i_c,i_r]
                i_f += 1
        ms.append(m)
    return ms


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
    if not __is_valid_matrix(m):
        raise WrongVarianceCovarianceMatrixException(m)
    chol_factorization = tf.linalg.cholesky(m)
    flat_chol_factorized_matrix = tfp.math.fill_triangular_inverse(chol_factorization)
    return flat_chol_factorized_matrix.numpy()


def spiral_flat_from_progressive(shape):
    # flat spiral representation from (flat) progressive representation
    size = int(0.5*(math.sqrt(1+8*shape)-1))
    low = int(math.ceil(size/2))
    up = size-low
    flat_spiral_indexes = []
    end = False
    c = 0
    l_index, u_index = None, None
    while not end:
        if c<low:
            l_index = (size-1)-c
            fl = ((l_index+1) * ((l_index+1) + 1)) // 2 -1
            for i in range(l_index, -1, -1):
                flat_spiral_indexes.append(fl+i-l_index)
        else:
            end = True
        if c<up:
            u_index = c
            fl = ((u_index+1) * ((u_index+1) + 1)) // 2 -1
            for i in range(0, u_index+1, +1):
                flat_spiral_indexes.append(fl-u_index+i)
        c+=1
    return flat_spiral_indexes


def __is_valid_matrix(matrix):
    if __is_positive_definite(matrix) and __is_symmetric(matrix):
        return True
    return False

def __is_positive_definite(matrix):
    return all(np.linalg.eigvals(matrix) > 0)

def __is_symmetric(matrix):
    return (matrix == matrix.T).all()


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


def normalize_value(value, original_low, original_upper, normal_low, normal_upper):
    if original_low == float('-inf'):
        original_low = -1000
    if original_upper == float('inf'):
        original_upper = 1000
    original_length = original_upper-original_low
    normal_length = normal_upper-normal_low
    factor = normal_length/original_length
    original_offset = value-original_low
    normalized_value = (original_offset)*factor+normal_low
    return normalized_value

def denormalize_value(normalized_value, original_low, original_upper, normal_low, normal_upper):
    if original_low == float('-inf'):
        original_low = -1000
    if original_upper == float('inf'):
        original_upper = 1000
    original_length = original_upper-original_low
    normal_length = normal_upper-normal_low
    factor = original_length/normal_length
    normal_offset = normalized_value-normal_low
    value = (normal_offset)*factor+original_low
    return value

def normalize_value_log(value, original_low, original_upper, normal_low, normal_upper):
    if original_low == float('-inf'):
        original_low = -1000
    if original_upper == float('inf'):
        original_upper = 1000
    original_length = math.log(original_upper)-math.log(original_low)
    normal_length = normal_upper-normal_low
    factor = normal_length/original_length
    original_offset = math.log(value)-math.log(original_low)
    normalized_value = (original_offset)*factor+normal_low
    return normalized_value

def denormalize_value_log(normalized_value, original_low, original_upper, normal_low, normal_upper):
    # it's the inverse function of '__normalize_value_log', after applying properties of logarithms
    if original_low == float('-inf'):
        original_low = -1000
    if original_upper == float('inf'):
        original_upper = 1000
    original_length = math.log(original_upper)-math.log(original_low)
    normal_length = normal_upper-normal_low
    factor = original_length/normal_length
    normal_offset = normalized_value-normal_low
    value = (normal_offset)*factor+math.log(original_low)
    return math.exp(value)


def is_keras_optimizer(opt):
    return isinstance(opt, KerasOptimizer)

def is_nevergrad_optimizer(opt):
    return isinstance(opt, NevergradOptimizer)