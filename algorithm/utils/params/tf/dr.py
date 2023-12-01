class DomainRandomization_parameters():
    def __init__(self, mode="multivariate_normal", seed=None, factors=None, params=None):
        self.mode = mode
        self.seed = seed
        if factors is None:
            factors = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        self.factors = factors
        self.params = params


    def set_seed(self, seed):
        self.seed = seed

    def set_factors(self, factors):
        self.factors = factors

    def set_multivariatenormal_params(self, mean_vector, variancecovariance_matrix):
        self.mode = "multivariate_normal"
        self.mean_vector = mean_vector
        self.variancecovariance_matrix = variancecovariance_matrix

    def set_univariatenormal_params(self, means, variances):
        self.mode = "univariate_normal"
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
