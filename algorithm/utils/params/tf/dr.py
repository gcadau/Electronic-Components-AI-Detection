from algorithm.utils.params.exceptions import *

class DomainRandomization_parameters():
    def __init__(self, mode="multivariate normal", seed=None, factors=None, params=None):
        self.mode = mode
        if self.mode not in ["multivariate normal", "univariate normal", "uniform", "triangular"]:
            raise NotFoundDomainRandomizationModeException(self.mode)
        self.seed = seed
        self.factors = factors
        if self.mode=="multivariate normal":
            if params is not None:
                self.mean_vector = params[0]
                self.variancecovariance_matrix = params[1]
            else:
                self.mean_vector = None
                self.variancecovariance_matrix = None
        if self.mode=="univariate normal":
            if params is not None:
                self.means = params[0]
                self.variances = params[1]
            else:
                self.means = None
                self.variances = None
        if self.mode=="uniform":
            if params is not None:
                self.lowers = params[0]
                self.uppers = params[1]
            else:
                self.lowers = None
                self.uppers = None
        if self.mode=="triangular":
            if params is not None:
                self.lowers = params[0]
                self.modes = params[1]
                self.uppers = params[2]
            else:
                self.lowers = None
                self.modes = None
                self.uppers = None


    def set_seed(self, seed):
        self.seed = seed

    def set_factors(self, factors):
        self.factors = factors

    def set_multivariatenormal_params(self, mean_vector, variancecovariance_matrix):
        self.mode = "multivariate normal"
        self.mean_vector = mean_vector
        self.variancecovariance_matrix = variancecovariance_matrix

    def set_univariatenormal_params(self, means, variances):
        self.mode = "univariate normal"
        self.means = means
        self.variances = variances

    def set_uniform_params(self, lowers, uppers):
        self.mode = "uniform"
        self.lowers = lowers
        self.uppers = uppers

    def set_triangular_params(self, lowers, modes, uppers):
        self.mode = "triangular"
        self.lowers = lowers
        self.modes = modes
        self.uppers = uppers


    def get_parameters_list(self):
        return ['brightness', 'contrast', 'horizontally flip', 'vertically flip', 'hue', 'jpeg quality', 'saturation']
