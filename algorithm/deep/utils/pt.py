import functools
import math

import numpy as np
import torch

from ..exceptions import WrongVarianceCovarianceMatrixException
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

try:  
    from nevergrad.optimization import Optimizer as NevergradOptimizer
except ImportError:  # pragma: no cover - keeps the module importable without it
    NevergradOptimizer = None



@functools.lru_cache(maxsize=None)
def _tri_index_map(n, upper=False):
    m = n * (n + 1) // 2
    k = np.arange(n * n)
    if upper:
        # concat([x, reverse(x[n:])])
        src = np.where(k < m, k, 2 * m - 1 - k)
        keep = np.triu(np.ones((n, n), dtype=bool)).reshape(-1)
    else:
        # concat([x[n:], reverse(x)])
        src = np.where(k < m - n, n + k, m - 1 - (k - (m - n)))
        keep = np.tril(np.ones((n, n), dtype=bool)).reshape(-1)
    pos = np.nonzero(keep)[0]
    return torch.as_tensor(pos, dtype=torch.long), torch.as_tensor(src[pos], dtype=torch.long)


def _side(m):
    n = int(round(0.5 * (math.sqrt(1 + 8 * m) - 1)))
    if n * (n + 1) // 2 != m:
        raise ValueError(f"{m} is not a triangular number, cannot build a square matrix.")
    return n


def fill_triangular(x, upper=False):
    x = torch.as_tensor(x)
    n = _side(x.shape[-1])
    pos, vec = _tri_index_map(n, upper)
    out = x.new_zeros(x.shape[:-1] + (n * n,))
    out[..., pos] = x[..., vec]
    return out.reshape(x.shape[:-1] + (n, n))


def fill_triangular_inverse(x, upper=False):
    x = torch.as_tensor(x)
    n = x.shape[-1]
    pos, vec = _tri_index_map(n, upper)
    flat = x.reshape(x.shape[:-2] + (n * n,))
    out = x.new_zeros(x.shape[:-2] + (n * (n + 1) // 2,))
    out[..., vec] = flat[..., pos]
    return out


def scaleAndFlat_matrix(flat_m, dtype=np.float32):
    flat_m = np.asarray(flat_m)
    size = int(0.5 * (math.sqrt(1 + 8 * flat_m.shape[0]) - 1))
    m = np.zeros((size, size), dtype=dtype)
    i_f = 0
    for i_c in range(size):
        for i_r in range(i_c + 1):
            m[i_c, i_r] = flat_m[i_f]
            i_f += 1
    for i_r in range(size):
        for i_c in range(i_r, size):
            m[i_r, i_c] = m[i_c, i_r]
            i_f += 1
    if not is_valid_matrix(m):
        raise WrongVarianceCovarianceMatrixException(m)
    chol_factorization = torch.linalg.cholesky(torch.from_numpy(m))
    flat_chol_factorized_matrix = fill_triangular_inverse(chol_factorization)
    return flat_chol_factorized_matrix.numpy()



class Triangular(torch.nn.Module):

    def __init__(self, low, mode, high, tf_legacy=False, generator=None):
        super().__init__()
        self.low = low
        self.mode = mode
        self.high = high
        self.tf_legacy = tf_legacy
        self.generator = generator

    def sample(self, sample_shape=()):
        low = torch.as_tensor(self.low, dtype=torch.float32)
        mode = torch.as_tensor(self.mode, dtype=torch.float32)
        high = torch.as_tensor(self.high, dtype=torch.float32)
        shape = torch.Size(sample_shape) + torch.broadcast_shapes(low.shape, mode.shape, high.shape)

        u = torch.rand(shape, generator=self.generator)
        span = high - low
        c = (mode - low) / span

        if self.tf_legacy:
            return torch.where(u < c,
                               low + torch.rand(shape, generator=self.generator) * (mode - low),
                               mode + torch.rand(shape, generator=self.generator) * (high - mode))

        return torch.where(u < c,
                           low + torch.sqrt(u * span * (mode - low)),
                           high - torch.sqrt((1.0 - u) * span * (high - mode)))

    def forward(self, sample_shape=()):
        return self.sample(sample_shape)



class MinMaxNorm_ElementWise:

    def __init__(self, min_values, max_values):
        self.min_values = min_values
        self.max_values = max_values

    def __call__(self, w):
        w = torch.as_tensor(w)
        return torch.clamp(w,
                           min=torch.as_tensor(self.min_values, dtype=w.dtype, device=w.device),
                           max=torch.as_tensor(self.max_values, dtype=w.dtype, device=w.device))

    @torch.no_grad()
    def apply_to(self, parameter):
        parameter.copy_(self(parameter))
        return parameter

    def get_config(self):
        return {'min_values': self.min_values, 'max_values': self.max_values}



def is_torch_optimizer(opt):
    return isinstance(opt, torch.optim.Optimizer)


def is_nevergrad_optimizer(opt):
    if NevergradOptimizer is not None:
        return isinstance(opt, NevergradOptimizer)
    return type(opt).__module__.startswith("nevergrad")
