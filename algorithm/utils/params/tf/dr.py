from algorithm.utils.params.exceptions import NotFoundDomainRandomizationModeException

class DomainRandomization_parameters():
    def __init__(self, mode="multivariate normal", seed=None, factors=None, params=None, optimize=True, ranges=None):
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
            if ranges is not None:
                self.mean_vector__ranges = ranges[0]
                self.variancecovariance_matrix__ranges = ranges[1]
            else:
                self.mean_vector__ranges = None
                self.variancecovariance_matrix__ranges = None
        if self.mode=="univariate normal":
            if params is not None:
                self.means = params[0]
                self.variances = params[1]
            else:
                self.means = None
                self.variances = None
            if ranges is not None:
                self.means__ranges = ranges[0]
                self.variances__ranges = ranges[1]
            else:
                self.means__ranges = None
                self.variances__ranges = None
        if self.mode=="uniform":
            if params is not None:
                self.lowers = params[0]
                self.uppers = params[1]
            else:
                self.lowers = None
                self.uppers = None
            if ranges is not None:
                self.lowers__ranges = ranges[0]
                self.uppers__ranges = ranges[1]
            else:
                self.lowers__ranges = None
                self.uppers__ranges = None
        if self.mode=="triangular":
            if params is not None:
                self.lowers = params[0]
                self.modes = params[1]
                self.uppers = params[2]
            else:
                self.lowers = None
                self.modes = None
                self.uppers = None
            if ranges is not None:
                self.lowers__ranges = ranges[0]
                self.modes__ranges = ranges[1]
                self.uppers__ranges = ranges[2]
            else:
                self.lowers__ranges = None
                self.modes__ranges = None
                self.uppers__ranges = None
        self.optimize = optimize


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

    def set_multivariatenormal_ranges(self, mean_vector, variancecovariance_matrix):
        self.mode = "multivariate normal"
        self.mean_vector__ranges = mean_vector
        self.variancecovariance_matrix__ranges = variancecovariance_matrix

    def set_univariatenormal_ranges(self, means, variances):
        self.mode = "univariate normal"
        self.means__ranges = means
        self.variances__ranges = variances

    def set_uniform__ranges(self, lowers, uppers):
        self.mode = "uniform"
        self.lowers__ranges = lowers
        self.uppers__ranges = uppers

    def set_triangular__ranges(self, lowers, modes, uppers):
        self.mode = "triangular"
        self.lowers__ranges = lowers
        self.modes__ranges = modes
        self.uppers__ranges = uppers


    def get_parameters_list(self):
        return ['brightness', 'contrast', 'horizontally flip', 'vertically flip', 'hue', 'jpeg quality', 'saturation']
