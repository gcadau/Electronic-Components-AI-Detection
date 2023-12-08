class NotFoundOptimizerException(Exception):
    def __init__(self):
        self.message = f"Optimizers not initialized."
        super().__init__(self.message)


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
