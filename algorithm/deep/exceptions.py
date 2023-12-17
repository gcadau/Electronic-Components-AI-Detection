class NotFoundOptimizerException(Exception):
    def __init__(self):
        self.message = f"Optimizers not initialized."
        super().__init__(self.message)


class WrongVarianceCovarianceMatrixException(Exception):
    def __init__(self, matrix):
        self.matrix = matrix
        self.message = f"Not correct variance covariance matrix.\n{self.matrix}\nis not symmetric positive definite."
        super().__init__(self.message)