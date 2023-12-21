import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import nevergrad as ng
import numpy as np
from collections.abc import Iterable
from itertools import chain
from algorithm.deep.exceptions import NotFoundOptimizerException, NotImplementedOptimizerException
from algorithm.deep.utils import (set_parameters__ranges, set_parameters__initials,
                                  fill_matrix, scaleAndFlat_matrix, spiral_flat_from_progressive, iterate_over_elements_below_diagonal,
                                  Triangular,
                                  MinMaxNorm_ElementWise,
                                  normalize_value, denormalize_value, normalize_value_log, denormalize_value_log)
from algorithm.domain_randomization.optimization.tf import (
    r_uniform as r_uniform_opt,
    r_triangular as r_triangular_opt,
    r_univariatenormal as r_univariatenormal_opt,
    r_multivariatenormal as r_multivariatenormal_opt
)
from algorithm.domain_randomization.tf import r_uniform, r_triangular, r_univariatenormal, r_multivariatenormal
from keras.optimizers import Optimizer as KerasOptimizer
from nevergrad.optimization import Optimizer as NevergradOptimizer
from algorithm.deep.utils import is_keras_optimizer, is_nevergrad_optimizer


class ResNetBlock(keras.layers.Layer):
    def __init__(self, filters):
        super(ResNetBlock, self).__init__()
        filters1, filters2 = filters
        # first 3x3 conv
        self.conv2a = keras.layers.Conv2D(filters1, 3, activation='relu', padding='same')
        # second 3x3 conv
        self.conv2b = keras.layers.Conv2D(filters2, 3, activation='relu', padding='same')

    def call(self, inputs):
        x = self.conv2a(inputs)
        x = self.conv2b(x)
        x += inputs
        return x


class ResNet1(keras.Model):
    def __init__(self, n_classes, input_shape=(128, 128, 3), field='data', domain_randomization=None):
        super(ResNet1, self).__init__()
        self.branch1_conv_1 = keras.layers.Conv2D(32, 3, activation="relu", input_shape=input_shape)
        self.branch1_conv_2 = keras.layers.Conv2D(64, 3, activation="relu")
        self.branch1_maxpool = keras.layers.MaxPooling2D(3)
        self.branch1_block_1 = ResNetBlock((64, 64))
        self.branch1_block_2 = ResNetBlock((64, 64))
        self.branch1_conv_3 = keras.layers.Conv2D(64, 3, activation='relu')
        self.branch1_glopool = keras.layers.GlobalAveragePooling2D()
        self.branch1_dense_1 = keras.layers.Dense(256, activation="relu")
        self.branch1_do = keras.layers.Dropout(0.5)
        self.branch1_dense_2 = keras.layers.Dense(n_classes)

        inputs = keras.layers.Input(shape=input_shape)
        x = self.branch1_conv_1(inputs)
        x = self.branch1_conv_2(x)
        x = self.branch1_maxpool(x)
        x = self.branch1_block_1(x)
        x = self.branch1_block_2(x)
        x = self.branch1_conv_3(x)
        x = self.branch1_glopool(x)
        x = self.branch1_dense_1(x)
        x = self.branch1_do(x)
        self.branch1_y = self.branch1_dense_2(x)

        self.model_branch1 = Branch(inputs=inputs, outputs=self.branch1_y, name="Main Branch")
        self.model_branch1.optimizer = None

        if domain_randomization is not None:
            self.domain_randomization = True
            parameters_name = domain_randomization.get_parameters_list()
            self.normalized_space = {"lower": 0.0, "upper":4.0}

            if domain_randomization.optimize:
                self.optimize = True

                self.model_branch2 = Branch(name="Random Distribution parameters branch")

                if domain_randomization.mode == "uniform":
                    if domain_randomization.lowers__ranges is None:
                        self.lowers__ranges = set_parameters__ranges("uniform", "lowers")
                    else:
                        self.lowers__ranges = domain_randomization.lowers__ranges
                    if domain_randomization.uppers__ranges is None:
                        self.uppers__ranges = set_parameters__ranges("uniform", "uppers")
                    else:
                        self.uppers__ranges = domain_randomization.uppers__ranges
                    if domain_randomization.lowers__initials is None:
                        self.lowers__initials = set_parameters__initials("uniform", "lowers", self.lowers__ranges)
                    else:
                        self.lowers__initials = domain_randomization.lowers__initials
                    if domain_randomization.uppers__initials is None:
                        self.uppers__initials = set_parameters__initials("uniform", "uppers", self.uppers__ranges)
                    else:
                        self.uppers__initials = domain_randomization.uppers__initials

                    self.branch2_lowerA = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[0],
                            self.lowers__ranges[0][0],
                            self.lowers__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerB = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[1],
                            self.lowers__ranges[1][0],
                            self.lowers__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerC = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[2],
                            self.lowers__ranges[2][0],
                            self.lowers__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerD = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[3],
                            self.lowers__ranges[3][0],
                            self.lowers__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerE = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[4],
                            self.lowers__ranges[4][0],
                            self.lowers__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerF = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[5],
                            self.lowers__ranges[5][0],
                            self.lowers__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerG = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[6],
                            self.lowers__ranges[6][0],
                            self.lowers__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperA = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[0],
                            self.uppers__ranges[0][0],
                            self.uppers__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperB = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[1],
                            self.uppers__ranges[1][0],
                            self.uppers__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperC = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[2],
                            self.uppers__ranges[2][0],
                            self.uppers__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperD = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[3],
                            self.uppers__ranges[3][0],
                            self.uppers__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperE = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[4],
                            self.uppers__ranges[4][0],
                            self.uppers__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperF = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[5],
                            self.uppers__ranges[5][0],
                            self.uppers__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_upperG = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[6],
                            self.uppers__ranges[6][0],
                            self.uppers__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.model_branch2.add_variables([
                        "branch2_lowerA",
                        "branch2_lowerB",
                        "branch2_lowerC",
                        "branch2_lowerD",
                        "branch2_lowerE",
                        "branch2_lowerF",
                        "branch2_lowerG"
                    ]+[
                        "branch2_upperA",
                        "branch2_upperB",
                        "branch2_upperC",
                        "branch2_upperD",
                        "branch2_upperE",
                        "branch2_upperF",
                        "branch2_upperG"
                    ])
                elif domain_randomization.mode == "triangular":
                    if domain_randomization.lowers__ranges is None:
                        self.lowers__ranges = set_parameters__ranges("triangular", "lowers")
                    else:
                        self.lowers__ranges = domain_randomization.lowers__ranges
                    if domain_randomization.modes__ranges is None:
                        self.modes__ranges = set_parameters__ranges("triangular", "modes")
                    else:
                        self.modes__ranges = domain_randomization.modes__ranges
                    if domain_randomization.uppers__ranges is None:
                        self.uppers__ranges = set_parameters__ranges("triangular", "uppers")
                    else:
                        self.uppers__ranges = domain_randomization.uppers__ranges
                    if domain_randomization.lowers__initials is None:
                        self.lowers__initials = set_parameters__initials("triangular", "lowers", self.lowers__ranges)
                    else:
                        self.lowers__initials = domain_randomization.lowers__initials
                    if domain_randomization.modes__initials is None:
                        self.modes__initials = set_parameters__initials("triangular", "modes", self.modes__ranges)
                    else:
                        self.modes__initials = domain_randomization.modes__initials
                    if domain_randomization.uppers__initials is None:
                        self.uppers__initials = set_parameters__initials("triangular", "uppers", self.uppers__ranges)
                    else:
                        self.uppers__initials = domain_randomization.uppers__initials

                    self.branch2_lowerA = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[0],
                            self.lowers__ranges[0][0],
                            self.lowers__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerB = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[1],
                            self.lowers__ranges[1][0],
                            self.lowers__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerC = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[2],
                            self.lowers__ranges[2][0],
                            self.lowers__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerD = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[3],
                            self.lowers__ranges[3][0],
                            self.lowers__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerE = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[4],
                            self.lowers__ranges[4][0],
                            self.lowers__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerF = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[5],
                            self.lowers__ranges[5][0],
                            self.lowers__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerG = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[6],
                            self.lowers__ranges[6][0],
                            self.lowers__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_modeA = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[0],
                            self.modes__ranges[0][0],
                            self.modes__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeB = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[1],
                            self.modes__ranges[1][0],
                            self.modes__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeC = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[2],
                            self.modes__ranges[2][0],
                            self.modes__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeD = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[3],
                            self.modes__ranges[3][0],
                            self.modes__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeE = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[4],
                            self.modes__ranges[4][0],
                            self.modes__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeF = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[5],
                            self.modes__ranges[5][0],
                            self.modes__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"]
                    )
                    self.branch2_modeG = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[6],
                            self.modes__ranges[6][0],
                            self.modes__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"]
                    )
                    self.branch2_upperA = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[0],
                            self.uppers__ranges[0][0],
                            self.uppers__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperB = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[1],
                            self.uppers__ranges[1][0],
                            self.uppers__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperC = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[2],
                            self.uppers__ranges[2][0],
                            self.uppers__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperD = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[3],
                            self.uppers__ranges[3][0],
                            self.uppers__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperE = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[4],
                            self.uppers__ranges[4][0],
                            self.uppers__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperF = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[5],
                            self.uppers__ranges[5][0],
                            self.uppers__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_upperG = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[6],
                            self.uppers__ranges[6][0],
                            self.uppers__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.model_branch2.add_variables([
                        "branch2_lowerA",
                        "branch2_lowerB",
                        "branch2_lowerC",
                        "branch2_lowerD",
                        "branch2_lowerE",
                        "branch2_lowerF",
                        "branch2_lowerG"
                    ]+[
                        "branch2_modeA",
                        "branch2_modeB",
                        "branch2_modeC",
                        "branch2_modeD",
                        "branch2_modeE",
                        "branch2_modeF",
                        "branch2_modeG"
                    ]+[
                        "branch2_upperA",
                        "branch2_upperB",
                        "branch2_upperC",
                        "branch2_upperD",
                        "branch2_upperE",
                        "branch2_upperF",
                        "branch2_upperG"
                    ])
                elif domain_randomization.mode == "univariate normal":
                    if domain_randomization.means__ranges is None:
                        self.means__ranges = set_parameters__ranges("univariatenormal", "means")
                    else:
                        self.means__ranges = domain_randomization.means__ranges
                    if domain_randomization.variances__ranges is None:
                        self.variances__ranges = set_parameters__ranges("univariatenormal", "variances")
                    else:
                        self.variances__ranges = domain_randomization.variances__ranges
                    if domain_randomization.means__initials is None:
                        self.means__initials = set_parameters__initials("univariatenormal", "means", self.means__ranges)
                    else:
                        self.means__initials = domain_randomization.means__initials
                    if domain_randomization.variances__initials is None:
                        self.variances__initials = set_parameters__initials("univariatenormal", "variances", self.variances__ranges)
                    else:
                        self.variances__initials = domain_randomization.variances__initials

                    self.branch2_meanA = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[0],
                            self.means__ranges[0][0],
                            self.means__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanB = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[1],
                            self.means__ranges[1][0],
                            self.means__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanC = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[2],
                            self.means__ranges[2][0],
                            self.means__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanD = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[3],
                            self.means__ranges[3][0],
                            self.means__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanE = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[4],
                            self.means__ranges[4][0],
                            self.means__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanF = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[5],
                            self.means__ranges[5][0],
                            self.means__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_meanG = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[6],
                            self.means__ranges[6][0],
                            self.means__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_varianceA = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[0],
                            self.variances__ranges[0][0],
                            self.variances__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceB = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[1],
                            self.variances__ranges[1][0],
                            self.variances__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceC = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[2],
                            self.variances__ranges[2][0],
                            self.variances__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceD = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[3],
                            self.variances__ranges[3][0],
                            self.variances__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceE = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[4],
                            self.variances__ranges[4][0],
                            self.variances__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceF = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[5],
                            self.variances__ranges[5][0],
                            self.variances__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_varianceG = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[6],
                            self.variances__ranges[6][0],
                            self.variances__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.model_branch2.add_variables([
                        "branch2_meanA",
                        "branch2_meanB",
                        "branch2_meanC",
                        "branch2_meanD",
                        "branch2_meanE",
                        "branch2_meanF",
                        "branch2_meanG"
                    ]+[
                        "branch2_varianceA",
                        "branch2_varianceB",
                        "branch2_varianceC",
                        "branch2_varianceD",
                        "branch2_varianceE",
                        "branch2_varianceF",
                        "branch2_varianceG"
                    ])
                elif domain_randomization.mode == "multivariate normal":
                    if domain_randomization.mean_vector__ranges is None:
                        self.mean_vector__ranges = set_parameters__ranges("multivariatenormal", "mean vector")
                    else:
                        self.mean_vector__ranges = domain_randomization.mean_vector__ranges
                    if domain_randomization.variancecovariance_matrix__ranges is None:
                        self.variancecovariance_matrix__ranges = set_parameters__ranges("multivariatenormal", "variancecovariance_matrix")
                    else:
                        self.variancecovariance_matrix__ranges = domain_randomization.variancecovariance_matrix__ranges
                    #self.variancecovariance_matrix__ranges___diagonal =
                    # [self.variancecovariance_matrix__ranges[k] for k in [i * (i + 1) // 2 + j for i in range(len(parameters_name)) for j in range(i + 1) if i==j]]
                    #self.variancecovariance_matrix__ranges___offDiagonal =
                    # [el for k, el in enumerate(self.variancecovariance_matrix__ranges) if k not in [i * (i + 1) // 2 + j for i in range(len(parameters_name)) for j in range(i + 1) if i==j]]
                    if domain_randomization.mean_vector__initials is None:
                        self.mean_vector__initials = set_parameters__initials("multivariatenormal", "mean vector", self.mean_vector__ranges)
                    else:
                        self.mean_vector__initials = domain_randomization.mean_vector__initials
                    if domain_randomization.variancecovariance_matrix__initials is None:
                        self.variancecovariance_matrix__initials = set_parameters__initials("multivariatenormal", "variancecovariance_matrix", self.variancecovariance_matrix__ranges)
                    else:
                        self.variancecovariance_matrix__initials = domain_randomization.variancecovariance_matrix__initials
                    self.branch2_mean_vector = []
                    for i in range(len(self.mean_vector__initials)):
                        self.branch2_mean_vector.append(
                            ng.p.Scalar(
                                init=normalize_value(
                                    self.mean_vector__initials[i],
                                    self.mean_vector__ranges[i][0],
                                    self.mean_vector__ranges[i][1],
                                    self.normalized_space["lower"],
                                    self.normalized_space["upper"]
                                )
                            ).set_bounds(
                                lower=self.normalized_space["lower"],
                                upper=self.normalized_space["upper"]
                            )
                        )
                    self.branch2_variancecovariance_matrix = []
                    for (element, i) in zip(
                        scaleAndFlat_matrix(
                            np.array(self.variancecovariance_matrix__initials)
                            # cholesky factorization of the matrix and flatten (spiral) version
                        ),
                        spiral_flat_from_progressive(
                            len(self.variancecovariance_matrix__initials)
                        ) # mapping of indexes from progressive (flat) representation to spiral flat representation
                    ):
                        self.branch2_variancecovariance_matrix.append(
                            ng.p.Scalar(
                                init=normalize_value(
                                    element,
                                    self.variancecovariance_matrix__ranges[i][0],
                                    self.variancecovariance_matrix__ranges[i][1],
                                    self.normalized_space["lower"],
                                    self.normalized_space["upper"]
                                )
                            ).set_bounds(
                                lower=self.normalized_space["lower"],
                                upper=self.normalized_space["upper"]
                            )
                        )
                    self.model_branch2.add_variables("branch2_mean_vector")
                    self.model_branch2.add_variables("branch2_variancecovariance_matrix")

                self.model_branch2.optimizer = None


            if domain_randomization.mode == "uniform":
                if domain_randomization.optimize:
                    params = {}
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_uniform_opt.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_uniform_opt.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_uniform_opt.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_uniform_opt.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_uniform_opt.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_uniform_opt.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_uniform_opt.layers.RandomSaturation(**(params["saturation"]))
                else:
                    self.optimize = False
                    params = {option: {} for option in parameters_name}
                    for option in parameters_name:
                        if domain_randomization.lowers is not None:
                            params[option]["lower"] = domain_randomization.lowers[parameters_name.index(option)]
                        if domain_randomization.uppers is not None:
                            params[option]["upper"] = domain_randomization.uppers[parameters_name.index(option)]
                        if domain_randomization.factors is not None:
                            params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                        params[option]["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_uniform.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_uniform.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_uniform.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_uniform.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_uniform.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_uniform.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_uniform.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "triangular":
                if domain_randomization.optimize:
                    params = {}
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_triangular_opt.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_triangular_opt.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_triangular_opt.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_triangular_opt.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_triangular_opt.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_triangular_opt.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_triangular_opt.layers.RandomSaturation(**(params["saturation"]))
                else:
                    self.optimize = False
                    params = {option: {} for option in parameters_name}
                    for option in parameters_name:
                        if domain_randomization.lowers is not None:
                            params[option]["lower"] = domain_randomization.lowers[parameters_name.index(option)]
                        if domain_randomization.modes is not None:
                            params[option]["mode"] = domain_randomization.modes[parameters_name.index(option)]
                        if domain_randomization.uppers is not None:
                            params[option]["upper"] = domain_randomization.uppers[parameters_name.index(option)]
                        if domain_randomization.factors is not None:
                            params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                        params[option]["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_triangular.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_triangular.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_triangular.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_triangular.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_triangular.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_triangular.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_triangular.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "univariate normal":
                if domain_randomization.optimize:
                    params = {}
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_univariatenormal_opt.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_univariatenormal_opt.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_univariatenormal_opt.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_univariatenormal_opt.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_univariatenormal_opt.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_univariatenormal_opt.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_univariatenormal_opt.layers.RandomSaturation(**(params["saturation"]))
                else:
                    self.optimize = False
                    params = {option: {} for option in parameters_name}
                    for option in parameters_name:
                        if domain_randomization.means is not None:
                            params[option]["mean"] = domain_randomization.means[parameters_name.index(option)]
                        if domain_randomization.variances is not None:
                            params[option]["variance"] = domain_randomization.variances[parameters_name.index(option)]
                        if domain_randomization.factors is not None:
                            params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                        params[option]["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_univariatenormal.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_univariatenormal.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_univariatenormal.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_univariatenormal.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_univariatenormal.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_univariatenormal.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_univariatenormal.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "multivariate normal":
                params = {}
                if domain_randomization.optimize:
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_parameters = r_multivariatenormal_opt.layers.RandomParameters(**params)
                else:
                    self.optimize = False
                    params["mean_vector"] = domain_randomization.mean_vector
                    params["variancecovariance_matrix"] = domain_randomization.variancecovariance_matrix
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_parameters = r_multivariatenormal.layers.RandomParameters(**params)
        else:
            self.domain_randomization = False
        self.domain_randomization__mode = domain_randomization.mode

        self.field = field

        if self.optimize:
            print("-> 2 optimizers needed when calling 'model.compile'\n")

        print("'model.compile' parameters info:")
        print("\tclass name required for loss and optimizers (e.g.: "
              "keras.losses.categorical_crossentropy or "
              "keras.optimizers.Adam or "
              "ng.optimizers.CMA")
        print("\tany additional parameters required must be passed as a dictionary in the second element of the tuple "
              "(class_name, parameters) (e.g.: (keras.optimizers.Adam, {'learning_rate':1e-3})")
        print("\t'run_eagerly=True' suggested.")


    def call(self, inputs, training=False):
        try:
            data = inputs[self.field]
        except KeyError:
            data = inputs
        if self.domain_randomization:
            if self.optimize and training:
                try: # self.domain_randomization__mode == "multivariate normal"
                    dr = self.branch1_random_parameters
                    sampled_params = tfp.distributions.MultivariateNormalTriL(
                        loc=self.branch2_mean_vector,
                        scale_tril=tfp.math.fill_triangular(
                            self.branch2_variancecovariance_matrix
                        )
                    ).sample(sample_shape=(data.shape[0],)).numpy()
                    data = dr(data, values=sampled_params, rand=False)
                except AttributeError:
                    if self.domain_randomization__mode == "uniform":
                        sampled_paramA = tfp.distributions.Uniform(
                            low=self.branch2_lowerA,
                            high=self.branch2_upperA
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramB = tfp.distributions.Uniform(
                            low=self.branch2_lowerB,
                            high=self.branch2_upperB
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramC = tfp.distributions.Uniform(
                            low=self.branch2_lowerC,
                            high=self.branch2_upperC
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramD = tfp.distributions.Uniform(
                            low=self.branch2_lowerD,
                            high=self.branch2_upperD
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramE = tfp.distributions.Uniform(
                            low=self.branch2_lowerE,
                            high=self.branch2_upperE
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramF = tfp.distributions.Uniform(
                            low=self.branch2_lowerF,
                            high=self.branch2_upperF
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramG = tfp.distributions.Uniform(
                            low=self.branch2_lowerG,
                            high=self.branch2_upperG
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                    if self.domain_randomization__mode == "triangular":
                        sampled_paramA = Triangular(
                            low=self.branch2_lowerA,
                            mode=self.branch2_modeA,
                            high=self.branch2_upperA
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramB = Triangular(
                            low=self.branch2_lowerB,
                            mode=self.branch2_modeB,
                            high=self.branch2_upperB
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramC = Triangular(
                            low=self.branch2_lowerC,
                            mode=self.branch2_modeC,
                            high=self.branch2_upperC
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramD = Triangular(
                            low=self.branch2_lowerD,
                            mode=self.branch2_modeD,
                            high=self.branch2_upperD
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramE = Triangular(
                            low=self.branch2_lowerE,
                            mode=self.branch2_modeE,
                            high=self.branch2_upperE
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramF = Triangular(
                            low=self.branch2_lowerF,
                            mode=self.branch2_modeF,
                            high=self.branch2_upperF
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramG = Triangular(
                            low=self.branch2_lowerG,
                            mode=self.branch2_modeG,
                            high=self.branch2_upperG
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                    if self.domain_randomization__mode == "univariate normal":
                        sampled_paramA = tfp.distributions.Normal(
                            loc=self.branch2_meanA,
                            scale=self.branch2_varianceA
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramB = tfp.distributions.Normal(
                            loc=self.branch2_meanB,
                            scale=self.branch2_varianceB
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramC = tfp.distributions.Normal(
                            loc=self.branch2_meanC,
                            scale=self.branch2_varianceC
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramD = tfp.distributions.Normal(
                            loc=self.branch2_meanD,
                            scale=self.branch2_varianceD
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramE = tfp.distributions.Normal(
                            loc=self.branch2_meanE,
                            scale=self.branch2_varianceE
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramF = tfp.distributions.Normal(
                            loc=self.branch2_meanF,
                            scale=self.branch2_varianceF
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramG = tfp.distributions.Normal(
                            loc=self.branch2_meanG,
                            scale=self.branch2_varianceG
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                    data = self.branch1_random_brightness(data, value=sampled_paramA, rand=False)
                    data = self.branch1_random_contrast(data, value=sampled_paramB, rand=False)
                    data = self.branch1_random_horizontally_flip(data, value=sampled_paramC, rand=False)
                    data = self.branch1_random_vertically_flip(data, value=sampled_paramD, rand=False)
                    data = self.branch1_random_hue(data, value=sampled_paramE, rand=False)
                    data = self.branch1_random_jpeg_quality(data, value=sampled_paramF, rand=False)
                    data = self.branch1_random_saturation(data, value=sampled_paramG, rand=False)
            else:
                try:
                    dr = self.branch1_random_parameters
                    data = dr(data)
                except AttributeError:
                    data = self.branch1_random_brightness(data)
                    data = self.branch1_random_contrast(data)
                    data = self.branch1_random_horizontally_flip(data)
                    data = self.branch1_random_vertically_flip(data)
                    data = self.branch1_random_hue(data)
                    data = self.branch1_random_jpeg_quality(data)
                    data = self.branch1_random_saturation(data)
        x = self.branch1_conv_1(data)
        x = self.branch1_conv_2(x)
        x = self.branch1_maxpool(x)
        x = self.branch1_block_1(x)
        x = self.branch1_block_2(x)
        x = self.branch1_conv_3(x)
        x = self.branch1_glopool(x)
        x = self.branch1_dense_1(x)
        x = self.branch1_do(x)
        x = self.branch1_dense_2(x)
        return x

    def train_step(self, data):
        if self.model_branch1.optimizer is None:
            par = None
            try:
                if isinstance(self.optimizer[0], tuple):
                    opt, par = self.optimizer[0]
                else:
                    opt = self.optimizer[0]
            except TypeError:
                if isinstance(self.optimizer, tuple):
                    opt, par = self.optimizer
                else:
                    opt = self.optimizer
            if par is not None:
                self.model_branch1.optimizer = opt(**par)
            else:
                self.model_branch1.optimizer = opt()
            self.__update_parameters(self.model_branch1.optimizer)
        if self.domain_randomization and self.optimize and self.model_branch2.optimizer is None:
            par = None
            if isinstance(self.optimizer[1], tuple):
                opt, par = self.optimizer[1]
            else:
                opt = self.optimizer[1]
            if par is None:
                par = {}
            trainableVars = [getattr(self, attr) for attr in self.model_branch2.get_trainable_variables()]
            if self.domain_randomization__mode != "multivariate normal":
                params = ng.p.Tuple(*trainableVars)
            else:
                params = ng.p.Tuple(*(list(chain(*trainableVars))))
            instrumentation = ng.p.Instrumentation(params=params)
            par["parametrization"] = instrumentation
            self.model_branch2.optimizer = opt(**par)
            self.__update_parameters(self.model_branch2.optimizer)
        if self.model_branch1.optimizer is None or ((self.domain_randomization and self.optimize) and self.model_branch2.optimizer is None):
            raise NotFoundOptimizerException()
        imgs, labs = data
        with tf.GradientTape(persistent=True) as tape:
            predictions = self(imgs, training=True)
            loss = self.compiled_loss(labs, predictions)
        loss_branch1 = loss
        if self.domain_randomization and self.optimize:
            if self.domain_randomization__mode == "uniform":
                # constraints: lowers<=uppers
                p = [[self.branch2_lowerA, self.branch2_upperA],
                     [self.branch2_lowerB, self.branch2_upperB],
                     [self.branch2_lowerC, self.branch2_upperC],
                     [self.branch2_lowerD, self.branch2_upperD],
                     [self.branch2_lowerE, self.branch2_upperE],
                     [self.branch2_lowerF, self.branch2_upperF],
                     [self.branch2_lowerG, self.branch2_upperG]
                     ]
                epsilon = 0.001
                ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.lowers__ranges, self.uppers__ranges)])
                width = ranges[:,1]-ranges[:,0]
                width = tf.where(tf.math.is_inf(width), 1000.0, width)
                rescaling_factor = width/(tf.reduce_max(width))
                penalty = tf.reduce_mean([ tf.maximum(tf.constant(0, dtype=tf.float32),
                                                      pi[0]-pi[1]+epsilon)
                                           /rescaling_factor[i]
                                           for i, pi in enumerate(p) ])
                loss_branch2 = loss+penalty
            if self.domain_randomization__mode == "triangular":
                # constraints: lowers<=modes, modes<=uppers
                p = [[self.branch2_lowerA, self.branch2_modeA, self.branch2_upperA],
                     [self.branch2_lowerB, self.branch2_modeB, self.branch2_upperA],
                     [self.branch2_lowerC, self.branch2_modeC, self.branch2_upperA],
                     [self.branch2_lowerD, self.branch2_modeD, self.branch2_upperA],
                     [self.branch2_lowerE, self.branch2_modeE, self.branch2_upperA],
                     [self.branch2_lowerF, self.branch2_modeF, self.branch2_upperA],
                     [self.branch2_lowerG, self.branch2_modeG, self.branch2_upperA]
                     ]
                epsilon = 0.001
                ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.lowers__ranges, self.modes__ranges)])
                width = ranges[:,1]-ranges[:,0]
                width = tf.where(tf.math.is_inf(width), 1000.0, width)
                rescaling_factor = width/(tf.reduce_max(width))
                penalty = tf.reduce_mean([ tf.maximum(tf.constant(0, dtype=tf.float32),
                                                      pi[0]-pi[1]+epsilon)
                                           /rescaling_factor[i]
                                           for i, pi in enumerate(p) ])
                loss_branch2 = loss+penalty
                ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.modes__ranges, self.uppers__ranges)])
                width = ranges[:,1]-ranges[:,0]
                width = tf.where(tf.math.is_inf(width), 1000.0, width)
                rescaling_factor = width/(tf.reduce_max(width))
                penalty = tf.reduce_mean([ tf.maximum(tf.constant(0, dtype=tf.float32),
                                                      pi[1]-pi[2]+epsilon)
                                           /rescaling_factor[i]
                                           for i, pi in enumerate(p) ])
                loss_branch2 += penalty
        self.__update_parameters(self.model_branch1.optimizer, loss_branch1, tape)
        if self.domain_randomization and self.optimize:
            self.__update_parameters(self.model_branch2.optimizer, loss_branch2)
        del tape
        return {"loss": loss}


    def __update_parameters(self, optimizer, loss=None, tape=None):
        if is_nevergrad_optimizer(optimizer):
            if loss is not None:
                optimizer.tell(self.__tmp_optimizer_branch2_x, loss)
            x = optimizer.ask()
            parameters = list(x.kwargs['params'])
            if self.domain_randomization__mode == "uniform":
                for i in range(len(self.lowers__initials)):
                    p = parameters[i]
                    p = denormalize_value(
                        p,
                        self.lowers__ranges[i][0],
                        self.lowers__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i] = p
                for i in range(len(self.uppers__initials)):
                    p = parameters[i+len(self.lowers__initials)]
                    p = denormalize_value(
                        p,
                        self.uppers__ranges[i][0],
                        self.uppers__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i+len(self.lowers__initials)] = p
            elif self.domain_randomization__mode == "triangular":
                for i in range(len(self.lowers__initials)):
                    p = parameters[i]
                    p = denormalize_value(
                        p,
                        self.lowers__ranges[i][0],
                        self.lowers__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i] = p
                for i in range(len(self.modes__initials)):
                    p = parameters[i+len(self.lowers__initials)]
                    p = denormalize_value(
                        p,
                        self.modes__ranges[i][0],
                        self.modes__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i+len(self.lowers__initials)] = p
                for i in range(len(self.uppers__initials)):
                    p = parameters[i+len(self.modes__initials)]
                    p = denormalize_value(
                        p,
                        self.uppers__ranges[i][0],
                        self.uppers__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i+len(self.modes__initials)] = p
            elif self.domain_randomization__mode == "multivariate normal":
                for i in range(len(self.mean_vector__initials)):
                    p = parameters[i]
                    p = denormalize_value(
                        p,
                        self.mean_vector__ranges[i][0],
                        self.mean_vector__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i] = p
                for i in spiral_flat_from_progressive(
                        len(self.variancecovariance_matrix__initials)
                ): # mapping of indexes from progressive (flat) representation to spiral flat representation
                    p = parameters[i+len(self.mean_vector__initials)]
                    p = denormalize_value(
                        p,
                        self.variancecovariance_matrix__ranges[i][0],
                        self.variancecovariance_matrix__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i+len(self.mean_vector__initials)] = p
            if self.domain_randomization__mode != "multivariate normal":
                for var, new_var in zip(self.model_branch2.get_trainable_variables(), parameters):
                    setattr(self, var, new_var)
            else:
                mv = parameters[0:
                                len(self.mean_vector__initials)]
                vcv = parameters[len(self.mean_vector__initials):
                                 len(self.mean_vector__initials)+len(self.variancecovariance_matrix__initials)]
                for var, new_var in zip(self.model_branch2.get_trainable_variables(), [mv, vcv]):
                    setattr(self, var, new_var)
            self.__tmp_optimizer_branch2_x = x
        elif is_keras_optimizer(optimizer):
            if tape is not None:
                gradients = tape.gradient(loss, self.model_branch1.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model_branch1.trainable_variables))
        else:
            raise NotImplementedOptimizerException(optimizer)




class ResNet2__0(keras.Model):
    def __init__(self, n_classes, input_shape=(128, 128, 3), field='data', domain_randomization=None):
        super(ResNet2__0, self).__init__()
        self.branch1_base_model = keras.applications.ResNet152(weights = 'imagenet', include_top = False, input_shape = input_shape)
        self.branch1_flatten = keras.layers.Flatten()
        self.branch1_dense1 = keras.layers.Dense(1000, activation='relu')
        self.branch1_dense2 = keras.layers.Dense(n_classes, activation='softmax')

        x = self.branch1_flatten(self.branch1_base_model.output)
        x = self.branch1_dense1(x)
        self.branch1_y = self.branch1_dense2(x)

        self.model_branch1 = Branch(inputs=self.branch1_base_model.input, outputs=self.branch1_y, name="Main Branch")
        self.model_branch1.optimizer = None

        if domain_randomization is not None:
            self.domain_randomization = True
            parameters_name = domain_randomization.get_parameters_list()
            self.normalized_space = {"lower": 0.0, "upper":4.0}

            if domain_randomization.optimize:
                self.optimize = True

                self.model_branch2 = Branch(name="Random Distribution parameters branch")

                if domain_randomization.mode == "uniform":
                    if domain_randomization.lowers__ranges is None:
                        self.lowers__ranges = set_parameters__ranges("uniform", "lowers")
                    else:
                        self.lowers__ranges = domain_randomization.lowers__ranges
                    if domain_randomization.uppers__ranges is None:
                        self.uppers__ranges = set_parameters__ranges("uniform", "uppers")
                    else:
                        self.uppers__ranges = domain_randomization.uppers__ranges
                    if domain_randomization.lowers__initials is None:
                        self.lowers__initials = set_parameters__initials("uniform", "lowers", self.lowers__ranges)
                    else:
                        self.lowers__initials = domain_randomization.lowers__initials
                    if domain_randomization.uppers__initials is None:
                        self.uppers__initials = set_parameters__initials("uniform", "uppers", self.uppers__ranges)
                    else:
                        self.uppers__initials = domain_randomization.uppers__initials

                    self.branch2_lowerA = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[0],
                            self.lowers__ranges[0][0],
                            self.lowers__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerB = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[1],
                            self.lowers__ranges[1][0],
                            self.lowers__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerC = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[2],
                            self.lowers__ranges[2][0],
                            self.lowers__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerD = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[3],
                            self.lowers__ranges[3][0],
                            self.lowers__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerE = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[4],
                            self.lowers__ranges[4][0],
                            self.lowers__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerF = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[5],
                            self.lowers__ranges[5][0],
                            self.lowers__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerG = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[6],
                            self.lowers__ranges[6][0],
                            self.lowers__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperA = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[0],
                            self.uppers__ranges[0][0],
                            self.uppers__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperB = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[1],
                            self.uppers__ranges[1][0],
                            self.uppers__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperC = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[2],
                            self.uppers__ranges[2][0],
                            self.uppers__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperD = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[3],
                            self.uppers__ranges[3][0],
                            self.uppers__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperE = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[4],
                            self.uppers__ranges[4][0],
                            self.uppers__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperF = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[5],
                            self.uppers__ranges[5][0],
                            self.uppers__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_upperG = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[6],
                            self.uppers__ranges[6][0],
                            self.uppers__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.model_branch2.add_variables([
                        "branch2_lowerA",
                        "branch2_lowerB",
                        "branch2_lowerC",
                        "branch2_lowerD",
                        "branch2_lowerE",
                        "branch2_lowerF",
                        "branch2_lowerG"
                    ]+[
                        "branch2_upperA",
                        "branch2_upperB",
                        "branch2_upperC",
                        "branch2_upperD",
                        "branch2_upperE",
                        "branch2_upperF",
                        "branch2_upperG"
                    ])
                elif domain_randomization.mode == "triangular":
                    if domain_randomization.lowers__ranges is None:
                        self.lowers__ranges = set_parameters__ranges("triangular", "lowers")
                    else:
                        self.lowers__ranges = domain_randomization.lowers__ranges
                    if domain_randomization.modes__ranges is None:
                        self.modes__ranges = set_parameters__ranges("triangular", "modes")
                    else:
                        self.modes__ranges = domain_randomization.modes__ranges
                    if domain_randomization.uppers__ranges is None:
                        self.uppers__ranges = set_parameters__ranges("triangular", "uppers")
                    else:
                        self.uppers__ranges = domain_randomization.uppers__ranges
                    if domain_randomization.lowers__initials is None:
                        self.lowers__initials = set_parameters__initials("triangular", "lowers", self.lowers__ranges)
                    else:
                        self.lowers__initials = domain_randomization.lowers__initials
                    if domain_randomization.modes__initials is None:
                        self.modes__initials = set_parameters__initials("triangular", "modes", self.modes__ranges)
                    else:
                        self.modes__initials = domain_randomization.modes__initials
                    if domain_randomization.uppers__initials is None:
                        self.uppers__initials = set_parameters__initials("triangular", "uppers", self.uppers__ranges)
                    else:
                        self.uppers__initials = domain_randomization.uppers__initials

                    self.branch2_lowerA = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[0],
                            self.lowers__ranges[0][0],
                            self.lowers__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerB = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[1],
                            self.lowers__ranges[1][0],
                            self.lowers__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerC = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[2],
                            self.lowers__ranges[2][0],
                            self.lowers__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerD = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[3],
                            self.lowers__ranges[3][0],
                            self.lowers__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerE = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[4],
                            self.lowers__ranges[4][0],
                            self.lowers__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerF = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[5],
                            self.lowers__ranges[5][0],
                            self.lowers__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerG = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[6],
                            self.lowers__ranges[6][0],
                            self.lowers__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_modeA = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[0],
                            self.modes__ranges[0][0],
                            self.modes__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeB = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[1],
                            self.modes__ranges[1][0],
                            self.modes__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeC = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[2],
                            self.modes__ranges[2][0],
                            self.modes__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeD = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[3],
                            self.modes__ranges[3][0],
                            self.modes__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeE = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[4],
                            self.modes__ranges[4][0],
                            self.modes__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeF = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[5],
                            self.modes__ranges[5][0],
                            self.modes__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"]
                    )
                    self.branch2_modeG = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[6],
                            self.modes__ranges[6][0],
                            self.modes__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"]
                    )
                    self.branch2_upperA = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[0],
                            self.uppers__ranges[0][0],
                            self.uppers__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperB = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[1],
                            self.uppers__ranges[1][0],
                            self.uppers__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperC = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[2],
                            self.uppers__ranges[2][0],
                            self.uppers__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperD = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[3],
                            self.uppers__ranges[3][0],
                            self.uppers__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperE = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[4],
                            self.uppers__ranges[4][0],
                            self.uppers__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperF = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[5],
                            self.uppers__ranges[5][0],
                            self.uppers__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_upperG = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[6],
                            self.uppers__ranges[6][0],
                            self.uppers__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.model_branch2.add_variables([
                        "branch2_lowerA",
                        "branch2_lowerB",
                        "branch2_lowerC",
                        "branch2_lowerD",
                        "branch2_lowerE",
                        "branch2_lowerF",
                        "branch2_lowerG"
                    ]+[
                        "branch2_modeA",
                        "branch2_modeB",
                        "branch2_modeC",
                        "branch2_modeD",
                        "branch2_modeE",
                        "branch2_modeF",
                        "branch2_modeG"
                    ]+[
                        "branch2_upperA",
                        "branch2_upperB",
                        "branch2_upperC",
                        "branch2_upperD",
                        "branch2_upperE",
                        "branch2_upperF",
                        "branch2_upperG"
                    ])
                elif domain_randomization.mode == "univariate normal":
                    if domain_randomization.means__ranges is None:
                        self.means__ranges = set_parameters__ranges("univariatenormal", "means")
                    else:
                        self.means__ranges = domain_randomization.means__ranges
                    if domain_randomization.variances__ranges is None:
                        self.variances__ranges = set_parameters__ranges("univariatenormal", "variances")
                    else:
                        self.variances__ranges = domain_randomization.variances__ranges
                    if domain_randomization.means__initials is None:
                        self.means__initials = set_parameters__initials("univariatenormal", "means", self.means__ranges)
                    else:
                        self.means__initials = domain_randomization.means__initials
                    if domain_randomization.variances__initials is None:
                        self.variances__initials = set_parameters__initials("univariatenormal", "variances", self.variances__ranges)
                    else:
                        self.variances__initials = domain_randomization.variances__initials

                    self.branch2_meanA = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[0],
                            self.means__ranges[0][0],
                            self.means__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanB = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[1],
                            self.means__ranges[1][0],
                            self.means__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanC = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[2],
                            self.means__ranges[2][0],
                            self.means__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanD = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[3],
                            self.means__ranges[3][0],
                            self.means__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanE = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[4],
                            self.means__ranges[4][0],
                            self.means__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanF = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[5],
                            self.means__ranges[5][0],
                            self.means__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_meanG = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[6],
                            self.means__ranges[6][0],
                            self.means__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_varianceA = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[0],
                            self.variances__ranges[0][0],
                            self.variances__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceB = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[1],
                            self.variances__ranges[1][0],
                            self.variances__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceC = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[2],
                            self.variances__ranges[2][0],
                            self.variances__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceD = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[3],
                            self.variances__ranges[3][0],
                            self.variances__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceE = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[4],
                            self.variances__ranges[4][0],
                            self.variances__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceF = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[5],
                            self.variances__ranges[5][0],
                            self.variances__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_varianceG = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[6],
                            self.variances__ranges[6][0],
                            self.variances__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.model_branch2.add_variables([
                        "branch2_meanA",
                        "branch2_meanB",
                        "branch2_meanC",
                        "branch2_meanD",
                        "branch2_meanE",
                        "branch2_meanF",
                        "branch2_meanG"
                    ]+[
                        "branch2_varianceA",
                        "branch2_varianceB",
                        "branch2_varianceC",
                        "branch2_varianceD",
                        "branch2_varianceE",
                        "branch2_varianceF",
                        "branch2_varianceG"
                    ])
                elif domain_randomization.mode == "multivariate normal":
                    if domain_randomization.mean_vector__ranges is None:
                        self.mean_vector__ranges = set_parameters__ranges("multivariatenormal", "mean vector")
                    else:
                        self.mean_vector__ranges = domain_randomization.mean_vector__ranges
                    if domain_randomization.variancecovariance_matrix__ranges is None:
                        self.variancecovariance_matrix__ranges = set_parameters__ranges("multivariatenormal", "variancecovariance_matrix")
                    else:
                        self.variancecovariance_matrix__ranges = domain_randomization.variancecovariance_matrix__ranges
                    #self.variancecovariance_matrix__ranges___diagonal =
                    # [self.variancecovariance_matrix__ranges[k] for k in [i * (i + 1) // 2 + j for i in range(len(parameters_name)) for j in range(i + 1) if i==j]]
                    #self.variancecovariance_matrix__ranges___offDiagonal =
                    # [el for k, el in enumerate(self.variancecovariance_matrix__ranges) if k not in [i * (i + 1) // 2 + j for i in range(len(parameters_name)) for j in range(i + 1) if i==j]]
                    if domain_randomization.mean_vector__initials is None:
                        self.mean_vector__initials = set_parameters__initials("multivariatenormal", "mean vector", self.mean_vector__ranges)
                    else:
                        self.mean_vector__initials = domain_randomization.mean_vector__initials
                    if domain_randomization.variancecovariance_matrix__initials is None:
                        self.variancecovariance_matrix__initials = set_parameters__initials("multivariatenormal", "variancecovariance_matrix", self.variancecovariance_matrix__ranges)
                    else:
                        self.variancecovariance_matrix__initials = domain_randomization.variancecovariance_matrix__initials
                    self.branch2_mean_vector = []
                    for i in range(len(self.mean_vector__initials)):
                        self.branch2_mean_vector.append(
                            ng.p.Scalar(
                                init=normalize_value(
                                    self.mean_vector__initials[i],
                                    self.mean_vector__ranges[i][0],
                                    self.mean_vector__ranges[i][1],
                                    self.normalized_space["lower"],
                                    self.normalized_space["upper"]
                                )
                            ).set_bounds(
                                lower=self.normalized_space["lower"],
                                upper=self.normalized_space["upper"]
                            )
                        )
                    self.branch2_variancecovariance_matrix = []
                    for (element, i) in zip(
                        scaleAndFlat_matrix(
                            np.array(self.variancecovariance_matrix__initials)
                            # cholesky factorization of the matrix and flatten (spiral) version
                        ),
                        spiral_flat_from_progressive(
                            len(self.variancecovariance_matrix__initials)
                        ) # mapping of indexes from progressive (flat) representation to spiral flat representation
                    ):
                        self.branch2_variancecovariance_matrix.append(
                            ng.p.Scalar(
                                init=normalize_value(
                                    element,
                                    self.variancecovariance_matrix__ranges[i][0],
                                    self.variancecovariance_matrix__ranges[i][1],
                                    self.normalized_space["lower"],
                                    self.normalized_space["upper"]
                                )
                            ).set_bounds(
                                lower=self.normalized_space["lower"],
                                upper=self.normalized_space["upper"]
                            )
                        )
                    self.model_branch2.add_variables("branch2_mean_vector")
                    self.model_branch2.add_variables("branch2_variancecovariance_matrix")

                self.model_branch2.optimizer = None


            if domain_randomization.mode == "uniform":
                if domain_randomization.optimize:
                    params = {}
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_uniform_opt.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_uniform_opt.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_uniform_opt.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_uniform_opt.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_uniform_opt.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_uniform_opt.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_uniform_opt.layers.RandomSaturation(**(params["saturation"]))
                else:
                    self.optimize = False
                    params = {option: {} for option in parameters_name}
                    for option in parameters_name:
                        if domain_randomization.lowers is not None:
                            params[option]["lower"] = domain_randomization.lowers[parameters_name.index(option)]
                        if domain_randomization.uppers is not None:
                            params[option]["upper"] = domain_randomization.uppers[parameters_name.index(option)]
                        if domain_randomization.factors is not None:
                            params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                        params[option]["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_uniform.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_uniform.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_uniform.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_uniform.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_uniform.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_uniform.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_uniform.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "triangular":
                if domain_randomization.optimize:
                    params = {}
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_triangular_opt.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_triangular_opt.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_triangular_opt.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_triangular_opt.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_triangular_opt.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_triangular_opt.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_triangular_opt.layers.RandomSaturation(**(params["saturation"]))
                else:
                    self.optimize = False
                    params = {option: {} for option in parameters_name}
                    for option in parameters_name:
                        if domain_randomization.lowers is not None:
                            params[option]["lower"] = domain_randomization.lowers[parameters_name.index(option)]
                        if domain_randomization.modes is not None:
                            params[option]["mode"] = domain_randomization.modes[parameters_name.index(option)]
                        if domain_randomization.uppers is not None:
                            params[option]["upper"] = domain_randomization.uppers[parameters_name.index(option)]
                        if domain_randomization.factors is not None:
                            params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                        params[option]["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_triangular.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_triangular.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_triangular.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_triangular.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_triangular.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_triangular.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_triangular.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "univariate normal":
                if domain_randomization.optimize:
                    params = {}
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_univariatenormal_opt.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_univariatenormal_opt.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_univariatenormal_opt.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_univariatenormal_opt.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_univariatenormal_opt.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_univariatenormal_opt.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_univariatenormal_opt.layers.RandomSaturation(**(params["saturation"]))
                else:
                    self.optimize = False
                    params = {option: {} for option in parameters_name}
                    for option in parameters_name:
                        if domain_randomization.means is not None:
                            params[option]["mean"] = domain_randomization.means[parameters_name.index(option)]
                        if domain_randomization.variances is not None:
                            params[option]["variance"] = domain_randomization.variances[parameters_name.index(option)]
                        if domain_randomization.factors is not None:
                            params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                        params[option]["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_univariatenormal.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_univariatenormal.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_univariatenormal.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_univariatenormal.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_univariatenormal.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_univariatenormal.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_univariatenormal.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "multivariate normal":
                params = {}
                if domain_randomization.optimize:
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_parameters = r_multivariatenormal_opt.layers.RandomParameters(**params)
                else:
                    self.optimize = False
                    params["mean_vector"] = domain_randomization.mean_vector
                    params["variancecovariance_matrix"] = domain_randomization.variancecovariance_matrix
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_parameters = r_multivariatenormal.layers.RandomParameters(**params)
        else:
            self.domain_randomization = False
        self.domain_randomization__mode = domain_randomization.mode

        self.field = field

        if self.optimize:
            print("-> 2 optimizers needed when calling 'model.compile'\n")

        print("'model.compile' parameters info:")
        print("\tclass name required for loss and optimizers (e.g.: "
              "keras.losses.categorical_crossentropy or "
              "keras.optimizers.Adam or "
              "ng.optimizers.CMA")
        print("\tany additional parameters required must be passed as a dictionary in the second element of the tuple "
              "(class_name, parameters) (e.g.: (keras.optimizers.Adam, {'learning_rate':1e-3})")
        print("\t'run_eagerly=True' suggested.")


    def call(self, inputs, training=False):
        try:
            data = inputs[self.field]
        except KeyError:
            data = inputs
        if self.domain_randomization:
            if self.optimize and training:
                try: # self.domain_randomization__mode == "multivariate normal"
                    dr = self.branch1_random_parameters
                    sampled_params = tfp.distributions.MultivariateNormalTriL(
                        loc=self.branch2_mean_vector,
                        scale_tril=tfp.math.fill_triangular(
                            self.branch2_variancecovariance_matrix
                        )
                    ).sample(sample_shape=(data.shape[0],)).numpy()
                    data = dr(data, values=sampled_params, rand=False)
                except AttributeError:
                    if self.domain_randomization__mode == "uniform":
                        sampled_paramA = tfp.distributions.Uniform(
                            low=self.branch2_lowerA,
                            high=self.branch2_upperA
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramB = tfp.distributions.Uniform(
                            low=self.branch2_lowerB,
                            high=self.branch2_upperB
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramC = tfp.distributions.Uniform(
                            low=self.branch2_lowerC,
                            high=self.branch2_upperC
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramD = tfp.distributions.Uniform(
                            low=self.branch2_lowerD,
                            high=self.branch2_upperD
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramE = tfp.distributions.Uniform(
                            low=self.branch2_lowerE,
                            high=self.branch2_upperE
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramF = tfp.distributions.Uniform(
                            low=self.branch2_lowerF,
                            high=self.branch2_upperF
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramG = tfp.distributions.Uniform(
                            low=self.branch2_lowerG,
                            high=self.branch2_upperG
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                    if self.domain_randomization__mode == "triangular":
                        sampled_paramA = Triangular(
                            low=self.branch2_lowerA,
                            mode=self.branch2_modeA,
                            high=self.branch2_upperA
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramB = Triangular(
                            low=self.branch2_lowerB,
                            mode=self.branch2_modeB,
                            high=self.branch2_upperB
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramC = Triangular(
                            low=self.branch2_lowerC,
                            mode=self.branch2_modeC,
                            high=self.branch2_upperC
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramD = Triangular(
                            low=self.branch2_lowerD,
                            mode=self.branch2_modeD,
                            high=self.branch2_upperD
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramE = Triangular(
                            low=self.branch2_lowerE,
                            mode=self.branch2_modeE,
                            high=self.branch2_upperE
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramF = Triangular(
                            low=self.branch2_lowerF,
                            mode=self.branch2_modeF,
                            high=self.branch2_upperF
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramG = Triangular(
                            low=self.branch2_lowerG,
                            mode=self.branch2_modeG,
                            high=self.branch2_upperG
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                    if self.domain_randomization__mode == "univariate normal":
                        sampled_paramA = tfp.distributions.Normal(
                            loc=self.branch2_meanA,
                            scale=self.branch2_varianceA
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramB = tfp.distributions.Normal(
                            loc=self.branch2_meanB,
                            scale=self.branch2_varianceB
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramC = tfp.distributions.Normal(
                            loc=self.branch2_meanC,
                            scale=self.branch2_varianceC
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramD = tfp.distributions.Normal(
                            loc=self.branch2_meanD,
                            scale=self.branch2_varianceD
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramE = tfp.distributions.Normal(
                            loc=self.branch2_meanE,
                            scale=self.branch2_varianceE
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramF = tfp.distributions.Normal(
                            loc=self.branch2_meanF,
                            scale=self.branch2_varianceF
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramG = tfp.distributions.Normal(
                            loc=self.branch2_meanG,
                            scale=self.branch2_varianceG
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                    data = self.branch1_random_brightness(data, value=sampled_paramA, rand=False)
                    data = self.branch1_random_contrast(data, value=sampled_paramB, rand=False)
                    data = self.branch1_random_horizontally_flip(data, value=sampled_paramC, rand=False)
                    data = self.branch1_random_vertically_flip(data, value=sampled_paramD, rand=False)
                    data = self.branch1_random_hue(data, value=sampled_paramE, rand=False)
                    data = self.branch1_random_jpeg_quality(data, value=sampled_paramF, rand=False)
                    data = self.branch1_random_saturation(data, value=sampled_paramG, rand=False)
            else:
                try:
                    dr = self.branch1_random_parameters
                    data = dr(data)
                except AttributeError:
                    data = self.branch1_random_brightness(data)
                    data = self.branch1_random_contrast(data)
                    data = self.branch1_random_horizontally_flip(data)
                    data = self.branch1_random_vertically_flip(data)
                    data = self.branch1_random_hue(data)
                    data = self.branch1_random_jpeg_quality(data)
                    data = self.branch1_random_saturation(data)
        base_model_output = self.branch1_base_model(data)
        x = self.branch1_flatten(base_model_output)
        x = self.branch1_dense1(x)
        y = self.branch1_dense2(x)
        return y

    def compile(self, optimizer, **kwargs):
        # optimizers configuration
        par = None
        try:
            if isinstance(optimizer[0], tuple):
                opt, par = optimizer[0]
            else:
                opt = optimizer[0]
        except TypeError:
            if isinstance(optimizer, tuple):
                opt, par = optimizer
            else:
                opt = optimizer
        if par is not None:
            self.model_branch1.optimizer = opt(**par)
        else:
            self.model_branch1.optimizer = opt()
        if self.domain_randomization and self.optimize:
            par = None
            if isinstance(optimizer[1], tuple):
                opt, par = optimizer[1]
            else:
                opt = optimizer[1]
            if par is None:
                par = {}
            trainableVars = [getattr(self, attr) for attr in self.model_branch2.get_trainable_variables()]
            if self.domain_randomization__mode != "multivariate normal":
                params = ng.p.Tuple(*trainableVars)
            else:
                params = ng.p.Tuple(*(list(chain(*trainableVars))))
            instrumentation = ng.p.Instrumentation(params=params)
            par["parametrization"] = instrumentation
            self.model_branch2.optimizer = opt(**par)
        if self.model_branch1.optimizer is None or ((self.domain_randomization and self.optimize) and self.model_branch2.optimizer is None):
            raise NotFoundOptimizerException()
        super(ResNet2__0, self).compile(optimizer=keras.optimizers.Adam(), **kwargs) # optimizers already handled: pass a 'default optimizer'

    def train_step(self, data):
        imgs, labs = data
        with tf.GradientTape(persistent=True) as tape:
            predictions = self(imgs, training=True)
            loss = self.compiled_loss(labs, predictions)
        loss_branch1 = loss
        loss_branch2 = loss
        if self.domain_randomization and self.optimize:
            if self.domain_randomization__mode == "uniform":
                # constraints: lowers<=uppers
                p = [[self.branch2_lowerA, self.branch2_upperA],
                     [self.branch2_lowerB, self.branch2_upperB],
                     [self.branch2_lowerC, self.branch2_upperC],
                     [self.branch2_lowerD, self.branch2_upperD],
                     [self.branch2_lowerE, self.branch2_upperE],
                     [self.branch2_lowerF, self.branch2_upperF],
                     [self.branch2_lowerG, self.branch2_upperG]
                     ]
                epsilon = 0.001
                ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.lowers__ranges, self.uppers__ranges)])
                width = ranges[:,1]-ranges[:,0]
                width = tf.where(tf.math.is_inf(width), 1000.0, width)
                rescaling_factor = width/(tf.reduce_max(width))
                penalty = tf.reduce_mean([ tf.maximum(tf.constant(0, dtype=tf.float32),
                                                      pi[0]-pi[1]+epsilon)
                                           /rescaling_factor[i]
                                           for i, pi in enumerate(p) ])
                loss_branch2 += penalty
            if self.domain_randomization__mode == "triangular":
                # constraints: lowers<=modes, modes<=uppers
                p = [[self.branch2_lowerA, self.branch2_modeA, self.branch2_upperA],
                     [self.branch2_lowerB, self.branch2_modeB, self.branch2_upperA],
                     [self.branch2_lowerC, self.branch2_modeC, self.branch2_upperA],
                     [self.branch2_lowerD, self.branch2_modeD, self.branch2_upperA],
                     [self.branch2_lowerE, self.branch2_modeE, self.branch2_upperA],
                     [self.branch2_lowerF, self.branch2_modeF, self.branch2_upperA],
                     [self.branch2_lowerG, self.branch2_modeG, self.branch2_upperA]
                     ]
                epsilon = 0.001
                ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.lowers__ranges, self.modes__ranges)])
                width = ranges[:,1]-ranges[:,0]
                width = tf.where(tf.math.is_inf(width), 1000.0, width)
                rescaling_factor = width/(tf.reduce_max(width))
                penalty = tf.reduce_mean([ tf.maximum(tf.constant(0, dtype=tf.float32),
                                                      pi[0]-pi[1]+epsilon)
                                           /rescaling_factor[i]
                                           for i, pi in enumerate(p) ])
                loss_branch2 += penalty
                ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.modes__ranges, self.uppers__ranges)])
                width = ranges[:,1]-ranges[:,0]
                width = tf.where(tf.math.is_inf(width), 1000.0, width)
                rescaling_factor = width/(tf.reduce_max(width))
                penalty = tf.reduce_mean([ tf.maximum(tf.constant(0, dtype=tf.float32),
                                                      pi[1]-pi[2]+epsilon)
                                           /rescaling_factor[i]
                                           for i, pi in enumerate(p) ])
                loss_branch2 += penalty
            if self.domain_randomization__mode == "multivariate normal":
                # constraints: elements of variance covariance matrix (complete version, not chol factorized ones)
                # between their ranges
                p = self.branch2_variancecovariance_matrix
                epsilon = 0.001
                sigma = tfp.math.fill_triangular(p)
                sigma = tf.linalg.matmul(sigma, sigma, transpose_b=True).numpy()
                width = tf.constant([self.variancecovariance_matrix__ranges[i][1]-self.variancecovariance_matrix__ranges[i][0]
                                      for i in range(len(self.variancecovariance_matrix__ranges))])
                width = tf.where(tf.math.is_inf(width), 1000.0, width)
                rescaling_factor = width/(tf.reduce_max(width))
                penalty = tf.reduce_mean([ tf.maximum(tf.constant(0, dtype=tf.float32),
                                                      sigma[i,j]-self.variancecovariance_matrix__ranges[index][1]+epsilon)
                                           /rescaling_factor[index]
                                           for index, (i,j) in enumerate(iterate_over_elements_below_diagonal(sigma.shape[0])) ])
                loss_branch2 += penalty
                penalty = tf.reduce_mean([ tf.maximum(tf.constant(0, dtype=tf.float32),
                                                      self.variancecovariance_matrix__ranges[index][0]-sigma[i,j]+epsilon)
                                           /rescaling_factor[index]
                                           for index, (i,j) in enumerate(iterate_over_elements_below_diagonal(sigma.shape[0])) ])
                loss_branch2 += penalty
        self.__update_parameters(self.model_branch1.optimizer, loss_branch1, tape)
        if self.domain_randomization and self.optimize:
            self.__update_parameters(self.model_branch2.optimizer, loss_branch2)
        del tape

        self.compiled_metrics.update_state(labs, predictions)
        return {'loss': loss, **{m.name: m.result() for m in self.metrics}}

    def fit(self, *args, fverbose=0, **kwargs):
        self.fverbose = fverbose
        if self.fverbose>0:
            try:
                if self.fverbose_path is None:
                    self.fverbose_path = "training_parameters.txt"
            except AttributeError:
                self.fverbose_path = "training_parameters.txt"
            self.fverbose_file = open(self.fverbose_path, 'w', encoding='utf8')
        # 1st update.
        self.__update_parameters(self.model_branch1.optimizer)
        if self.domain_randomization and self.optimize:
            self.__update_parameters(self.model_branch2.optimizer)
        super(ResNet2__0, self).fit(*args, **kwargs)
        if self.fverbose>0:
            self.fverbose_file.close()


    def __update_parameters(self, optimizer, loss=None, tape=None):
        if is_nevergrad_optimizer(optimizer):
            if loss is not None:
                optimizer.tell(self.__tmp_optimizer_branch2_x, float(loss.numpy()))
            x = optimizer.ask()
            parameters = list(x.kwargs['params'])
            if self.domain_randomization__mode == "uniform":
                for i in range(len(self.lowers__initials)):
                    p = parameters[i]
                    p = denormalize_value(
                        p,
                        self.lowers__ranges[i][0],
                        self.lowers__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i] = p
                for i in range(len(self.uppers__initials)):
                    p = parameters[i+len(self.lowers__initials)]
                    p = denormalize_value(
                        p,
                        self.uppers__ranges[i][0],
                        self.uppers__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i+len(self.lowers__initials)] = p
            elif self.domain_randomization__mode == "triangular":
                for i in range(len(self.lowers__initials)):
                    p = parameters[i]
                    p = denormalize_value(
                        p,
                        self.lowers__ranges[i][0],
                        self.lowers__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i] = p
                for i in range(len(self.modes__initials)):
                    p = parameters[i+len(self.lowers__initials)]
                    p = denormalize_value(
                        p,
                        self.modes__ranges[i][0],
                        self.modes__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i+len(self.lowers__initials)] = p
                for i in range(len(self.uppers__initials)):
                    p = parameters[i+len(self.modes__initials)]
                    p = denormalize_value(
                        p,
                        self.uppers__ranges[i][0],
                        self.uppers__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i+len(self.modes__initials)] = p
            elif self.domain_randomization__mode == "multivariate normal":
                for i in range(len(self.mean_vector__initials)):
                    p = parameters[i]
                    p = denormalize_value(
                        p,
                        self.mean_vector__ranges[i][0],
                        self.mean_vector__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i] = p
                for i in spiral_flat_from_progressive(
                        len(self.variancecovariance_matrix__initials)
                ): # mapping of indexes from progressive (flat) representation to spiral flat representation
                    p = parameters[i+len(self.mean_vector__initials)]
                    p = denormalize_value(
                        p,
                        self.variancecovariance_matrix__ranges[i][0],
                        self.variancecovariance_matrix__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i+len(self.mean_vector__initials)] = p
            if self.domain_randomization__mode != "multivariate normal":
                for var, new_var in zip(self.model_branch2.get_trainable_variables(), parameters):
                    setattr(self, var, new_var)
            else:
                mv = parameters[0:
                                len(self.mean_vector__initials)]
                vcv = parameters[len(self.mean_vector__initials):
                                 len(self.mean_vector__initials)+len(self.variancecovariance_matrix__initials)]
                for var, new_var in zip(self.model_branch2.get_trainable_variables(), [mv, vcv]):
                    setattr(self, var, new_var)
            self.__tmp_optimizer_branch2_x = x
            if self.fverbose==2 or self.fverbose==3:
                self.__print_fverbose(branch=2)
        elif is_keras_optimizer(optimizer):
            if tape is not None:
                gradients = tape.gradient(loss, self.model_branch1.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model_branch1.trainable_variables))
            if self.fverbose==1 or self.fverbose==3:
                self.__print_fverbose(branch=1)
        else:
            raise NotImplementedOptimizerException(optimizer)


    def set_fverbose__file_path(self, filepath):
        self.fverbose_path = filepath

    def get_fverbose__file_path(self):
        return self.fverbose_path


    def __print_fverbose(self, branch=2):
        if branch == 1:
            variables = self.model_branch1.trainable_variables
            for var in variables:
                self.fverbose_file.write(f"{var.name}: {var.numpy()}\n")
            self.fverbose_file.write("\n\n")
        if branch == 2:
            if self.domain_randomization__mode == "uniform":
                variables = [getattr(self, var) for var in self.model_branch2.get_trainable_variables()]
                n_params = len(self.lowers__initials)
                for i in range(n_params):
                    low = variables[i]
                    upper = variables[n_params+i]
                    self.fverbose_file.write(f"Param {i}: U({low}, {upper})\n")
            if self.domain_randomization__mode == "triangular":
                variables = [getattr(self, var) for var in self.model_branch2.get_trainable_variables()]
                n_params = len(self.lowers__initials)
                for i in range(n_params):
                    low = variables[i]
                    mode = variables[n_params+i]
                    upper = variables[n_params+n_params+i]
                    self.fverbose_file.write(f"Param {i}: Tr({low}, {mode}, {upper})\n")
            if self.domain_randomization__mode == "univariate normal":
                variables = [getattr(self, var) for var in self.model_branch2.get_trainable_variables()]
                n_params = len(self.means__initials)
                for i in range(n_params):
                    mean = variables[i]
                    variance = variables[n_params+i]
                    self.fverbose_file.write(f"Param {i}: N({mean}, {variance})\n")
            if self.domain_randomization__mode == "multivariate normal":
                variables = [getattr(self, var) for var in self.model_branch2.get_trainable_variables()]
                mu = np.array(variables[0])
                sigma = tfp.math.fill_triangular(variables[1])
                sigma = tf.linalg.matmul(sigma, sigma, transpose_b=True).numpy()
                self.fverbose_file.write(f"Params [1,...,i,...,n]ᵀ: N(mu, SIGMA)\n")
                self.fverbose_file.write(f"\tmu = {mu}\n")
                self.fverbose_file.write(f"\tSIGMA = ")
                c = 1
                for row in sigma:
                    if c==1:
                        self.fverbose_file.write("[" + " ".join(map(str, row)) + "\n")
                    elif c==sigma.shape[0]:
                        self.fverbose_file.write("\t         " + " ".join(map(str, row)) + "]\n")
                    else:
                        self.fverbose_file.write("\t         " + " ".join(map(str, row)) + "\n")
                    c+=1
            self.fverbose_file.write("\n\n")




class ResNet2__1(keras.Model):
    def __init__(self, n_classes, input_shape=(128, 128, 3), field='data', domain_randomization=None):
        super(ResNet2__1, self).__init__()
        self.branch1_base_model = keras.applications.ResNet152(weights = 'imagenet', include_top = False, input_shape = input_shape)
        for layer in self.base_model.layers:
            layer.trainable = False
        self.branch1_flatten = keras.layers.Flatten()
        self.branch1_dense1 = keras.layers.Dense(1000, activation='relu')
        self.branch1_dense2 = keras.layers.Dense(n_classes, activation='softmax')


        x = self.branch1_flatten(self.branch1_base_model.output)
        x = self.branch1_dense1(x)
        self.branch1_y = self.branch1_dense2(x)

        self.model_branch1 = Branch(inputs=self.branch1_base_model.input, outputs=self.branch1_y, name="Main Branch")
        self.model_branch1.optimizer = None

        if domain_randomization is not None:
            self.domain_randomization = True
            parameters_name = domain_randomization.get_parameters_list()
            self.normalized_space = {"lower": 0.0, "upper":4.0}

            if domain_randomization.optimize:
                self.optimize = True

                self.model_branch2 = Branch(name="Random Distribution parameters branch")

                if domain_randomization.mode == "uniform":
                    if domain_randomization.lowers__ranges is None:
                        self.lowers__ranges = set_parameters__ranges("uniform", "lowers")
                    else:
                        self.lowers__ranges = domain_randomization.lowers__ranges
                    if domain_randomization.uppers__ranges is None:
                        self.uppers__ranges = set_parameters__ranges("uniform", "uppers")
                    else:
                        self.uppers__ranges = domain_randomization.uppers__ranges
                    if domain_randomization.lowers__initials is None:
                        self.lowers__initials = set_parameters__initials("uniform", "lowers", self.lowers__ranges)
                    else:
                        self.lowers__initials = domain_randomization.lowers__initials
                    if domain_randomization.uppers__initials is None:
                        self.uppers__initials = set_parameters__initials("uniform", "uppers", self.uppers__ranges)
                    else:
                        self.uppers__initials = domain_randomization.uppers__initials

                    self.branch2_lowerA = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[0],
                            self.lowers__ranges[0][0],
                            self.lowers__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerB = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[1],
                            self.lowers__ranges[1][0],
                            self.lowers__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerC = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[2],
                            self.lowers__ranges[2][0],
                            self.lowers__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerD = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[3],
                            self.lowers__ranges[3][0],
                            self.lowers__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerE = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[4],
                            self.lowers__ranges[4][0],
                            self.lowers__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerF = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[5],
                            self.lowers__ranges[5][0],
                            self.lowers__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerG = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[6],
                            self.lowers__ranges[6][0],
                            self.lowers__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperA = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[0],
                            self.uppers__ranges[0][0],
                            self.uppers__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperB = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[1],
                            self.uppers__ranges[1][0],
                            self.uppers__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperC = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[2],
                            self.uppers__ranges[2][0],
                            self.uppers__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperD = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[3],
                            self.uppers__ranges[3][0],
                            self.uppers__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperE = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[4],
                            self.uppers__ranges[4][0],
                            self.uppers__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperF = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[5],
                            self.uppers__ranges[5][0],
                            self.uppers__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_upperG = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[6],
                            self.uppers__ranges[6][0],
                            self.uppers__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.model_branch2.add_variables([
                        "branch2_lowerA",
                        "branch2_lowerB",
                        "branch2_lowerC",
                        "branch2_lowerD",
                        "branch2_lowerE",
                        "branch2_lowerF",
                        "branch2_lowerG"
                    ]+[
                        "branch2_upperA",
                        "branch2_upperB",
                        "branch2_upperC",
                        "branch2_upperD",
                        "branch2_upperE",
                        "branch2_upperF",
                        "branch2_upperG"
                    ])
                elif domain_randomization.mode == "triangular":
                    if domain_randomization.lowers__ranges is None:
                        self.lowers__ranges = set_parameters__ranges("triangular", "lowers")
                    else:
                        self.lowers__ranges = domain_randomization.lowers__ranges
                    if domain_randomization.modes__ranges is None:
                        self.modes__ranges = set_parameters__ranges("triangular", "modes")
                    else:
                        self.modes__ranges = domain_randomization.modes__ranges
                    if domain_randomization.uppers__ranges is None:
                        self.uppers__ranges = set_parameters__ranges("triangular", "uppers")
                    else:
                        self.uppers__ranges = domain_randomization.uppers__ranges
                    if domain_randomization.lowers__initials is None:
                        self.lowers__initials = set_parameters__initials("triangular", "lowers", self.lowers__ranges)
                    else:
                        self.lowers__initials = domain_randomization.lowers__initials
                    if domain_randomization.modes__initials is None:
                        self.modes__initials = set_parameters__initials("triangular", "modes", self.modes__ranges)
                    else:
                        self.modes__initials = domain_randomization.modes__initials
                    if domain_randomization.uppers__initials is None:
                        self.uppers__initials = set_parameters__initials("triangular", "uppers", self.uppers__ranges)
                    else:
                        self.uppers__initials = domain_randomization.uppers__initials

                    self.branch2_lowerA = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[0],
                            self.lowers__ranges[0][0],
                            self.lowers__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerB = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[1],
                            self.lowers__ranges[1][0],
                            self.lowers__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerC = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[2],
                            self.lowers__ranges[2][0],
                            self.lowers__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerD = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[3],
                            self.lowers__ranges[3][0],
                            self.lowers__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerE = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[4],
                            self.lowers__ranges[4][0],
                            self.lowers__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerF = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[5],
                            self.lowers__ranges[5][0],
                            self.lowers__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_lowerG = ng.p.Scalar(
                        init=normalize_value(
                            self.lowers__initials[6],
                            self.lowers__ranges[6][0],
                            self.lowers__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_modeA = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[0],
                            self.modes__ranges[0][0],
                            self.modes__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeB = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[1],
                            self.modes__ranges[1][0],
                            self.modes__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeC = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[2],
                            self.modes__ranges[2][0],
                            self.modes__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeD = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[3],
                            self.modes__ranges[3][0],
                            self.modes__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeE = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[4],
                            self.modes__ranges[4][0],
                            self.modes__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"],
                    )
                    self.branch2_modeF = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[5],
                            self.modes__ranges[5][0],
                            self.modes__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"]
                    )
                    self.branch2_modeG = ng.p.Scalar(
                        init=normalize_value(
                            self.modes__initials[6],
                            self.modes__ranges[6][0],
                            self.modes__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["mode"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["mode"]
                    )
                    self.branch2_upperA = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[0],
                            self.uppers__ranges[0][0],
                            self.uppers__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperB = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[1],
                            self.uppers__ranges[1][0],
                            self.uppers__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperC = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[2],
                            self.uppers__ranges[2][0],
                            self.uppers__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperD = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[3],
                            self.uppers__ranges[3][0],
                            self.uppers__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperE = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[4],
                            self.uppers__ranges[4][0],
                            self.uppers__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_upperF = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[5],
                            self.uppers__ranges[5][0],
                            self.uppers__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_upperG = ng.p.Scalar(
                        init=normalize_value(
                            self.uppers__initials[6],
                            self.uppers__ranges[6][0],
                            self.uppers__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.model_branch2.add_variables([
                        "branch2_lowerA",
                        "branch2_lowerB",
                        "branch2_lowerC",
                        "branch2_lowerD",
                        "branch2_lowerE",
                        "branch2_lowerF",
                        "branch2_lowerG"
                    ]+[
                        "branch2_modeA",
                        "branch2_modeB",
                        "branch2_modeC",
                        "branch2_modeD",
                        "branch2_modeE",
                        "branch2_modeF",
                        "branch2_modeG"
                    ]+[
                        "branch2_upperA",
                        "branch2_upperB",
                        "branch2_upperC",
                        "branch2_upperD",
                        "branch2_upperE",
                        "branch2_upperF",
                        "branch2_upperG"
                    ])
                elif domain_randomization.mode == "univariate normal":
                    if domain_randomization.means__ranges is None:
                        self.means__ranges = set_parameters__ranges("univariatenormal", "means")
                    else:
                        self.means__ranges = domain_randomization.means__ranges
                    if domain_randomization.variances__ranges is None:
                        self.variances__ranges = set_parameters__ranges("univariatenormal", "variances")
                    else:
                        self.variances__ranges = domain_randomization.variances__ranges
                    if domain_randomization.means__initials is None:
                        self.means__initials = set_parameters__initials("univariatenormal", "means", self.means__ranges)
                    else:
                        self.means__initials = domain_randomization.means__initials
                    if domain_randomization.variances__initials is None:
                        self.variances__initials = set_parameters__initials("univariatenormal", "variances", self.variances__ranges)
                    else:
                        self.variances__initials = domain_randomization.variances__initials

                    self.branch2_meanA = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[0],
                            self.means__ranges[0][0],
                            self.means__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanB = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[1],
                            self.means__ranges[1][0],
                            self.means__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanC = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[2],
                            self.means__ranges[2][0],
                            self.means__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanD = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[3],
                            self.means__ranges[3][0],
                            self.means__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanE = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[4],
                            self.means__ranges[4][0],
                            self.means__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_meanF = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[5],
                            self.means__ranges[5][0],
                            self.means__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_meanG = ng.p.Scalar(
                        init=normalize_value(
                            self.means__initials[6],
                            self.means__ranges[6][0],
                            self.means__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_varianceA = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[0],
                            self.variances__ranges[0][0],
                            self.variances__ranges[0][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceB = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[1],
                            self.variances__ranges[1][0],
                            self.variances__ranges[1][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceC = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[2],
                            self.variances__ranges[2][0],
                            self.variances__ranges[2][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceD = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[3],
                            self.variances__ranges[3][0],
                            self.variances__ranges[3][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceE = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[4],
                            self.variances__ranges[4][0],
                            self.variances__ranges[4][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"],
                    )
                    self.branch2_varianceF = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[5],
                            self.variances__ranges[5][0],
                            self.variances__ranges[5][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.branch2_varianceG = ng.p.Scalar(
                        init=normalize_value(
                            self.variances__initials[6],
                            self.variances__ranges[6][0],
                            self.variances__ranges[6][1],
                            self.normalized_space["lower"],
                            self.normalized_space["upper"]
                        )
                    ).set_bounds(
                        lower=self.normalized_space["lower"],
                        upper=self.normalized_space["upper"]
                    )
                    self.model_branch2.add_variables([
                        "branch2_meanA",
                        "branch2_meanB",
                        "branch2_meanC",
                        "branch2_meanD",
                        "branch2_meanE",
                        "branch2_meanF",
                        "branch2_meanG"
                    ]+[
                        "branch2_varianceA",
                        "branch2_varianceB",
                        "branch2_varianceC",
                        "branch2_varianceD",
                        "branch2_varianceE",
                        "branch2_varianceF",
                        "branch2_varianceG"
                    ])
                elif domain_randomization.mode == "multivariate normal":
                    if domain_randomization.mean_vector__ranges is None:
                        self.mean_vector__ranges = set_parameters__ranges("multivariatenormal", "mean vector")
                    else:
                        self.mean_vector__ranges = domain_randomization.mean_vector__ranges
                    if domain_randomization.variancecovariance_matrix__ranges is None:
                        self.variancecovariance_matrix__ranges = set_parameters__ranges("multivariatenormal", "variancecovariance_matrix")
                    else:
                        self.variancecovariance_matrix__ranges = domain_randomization.variancecovariance_matrix__ranges
                    #self.variancecovariance_matrix__ranges___diagonal =
                    # [self.variancecovariance_matrix__ranges[k] for k in [i * (i + 1) // 2 + j for i in range(len(parameters_name)) for j in range(i + 1) if i==j]]
                    #self.variancecovariance_matrix__ranges___offDiagonal =
                    # [el for k, el in enumerate(self.variancecovariance_matrix__ranges) if k not in [i * (i + 1) // 2 + j for i in range(len(parameters_name)) for j in range(i + 1) if i==j]]
                    if domain_randomization.mean_vector__initials is None:
                        self.mean_vector__initials = set_parameters__initials("multivariatenormal", "mean vector", self.mean_vector__ranges)
                    else:
                        self.mean_vector__initials = domain_randomization.mean_vector__initials
                    if domain_randomization.variancecovariance_matrix__initials is None:
                        self.variancecovariance_matrix__initials = set_parameters__initials("multivariatenormal", "variancecovariance_matrix", self.variancecovariance_matrix__ranges)
                    else:
                        self.variancecovariance_matrix__initials = domain_randomization.variancecovariance_matrix__initials
                    self.branch2_mean_vector = []
                    for i in range(len(self.mean_vector__initials)):
                        self.branch2_mean_vector.append(
                            ng.p.Scalar(
                                init=normalize_value(
                                    self.mean_vector__initials[i],
                                    self.mean_vector__ranges[i][0],
                                    self.mean_vector__ranges[i][1],
                                    self.normalized_space["lower"],
                                    self.normalized_space["upper"]
                                )
                            ).set_bounds(
                                lower=self.normalized_space["lower"],
                                upper=self.normalized_space["upper"]
                            )
                        )
                    self.branch2_variancecovariance_matrix = []
                    for (element, i) in zip(
                        scaleAndFlat_matrix(
                            np.array(self.variancecovariance_matrix__initials)
                            # cholesky factorization of the matrix and flatten (spiral) version
                        ),
                        spiral_flat_from_progressive(
                            len(self.variancecovariance_matrix__initials)
                        ) # mapping of indexes from progressive (flat) representation to spiral flat representation
                    ):
                        self.branch2_variancecovariance_matrix.append(
                            ng.p.Scalar(
                                init=normalize_value(
                                    element,
                                    self.variancecovariance_matrix__ranges[i][0],
                                    self.variancecovariance_matrix__ranges[i][1],
                                    self.normalized_space["lower"],
                                    self.normalized_space["upper"]
                                )
                            ).set_bounds(
                                lower=self.normalized_space["lower"],
                                upper=self.normalized_space["upper"]
                            )
                        )
                    self.model_branch2.add_variables("branch2_mean_vector")
                    self.model_branch2.add_variables("branch2_variancecovariance_matrix")

                self.model_branch2.optimizer = None


            if domain_randomization.mode == "uniform":
                if domain_randomization.optimize:
                    params = {}
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_uniform_opt.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_uniform_opt.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_uniform_opt.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_uniform_opt.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_uniform_opt.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_uniform_opt.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_uniform_opt.layers.RandomSaturation(**(params["saturation"]))
                else:
                    self.optimize = False
                    params = {option: {} for option in parameters_name}
                    for option in parameters_name:
                        if domain_randomization.lowers is not None:
                            params[option]["lower"] = domain_randomization.lowers[parameters_name.index(option)]
                        if domain_randomization.uppers is not None:
                            params[option]["upper"] = domain_randomization.uppers[parameters_name.index(option)]
                        if domain_randomization.factors is not None:
                            params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                        params[option]["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_uniform.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_uniform.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_uniform.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_uniform.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_uniform.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_uniform.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_uniform.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "triangular":
                if domain_randomization.optimize:
                    params = {}
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_triangular_opt.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_triangular_opt.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_triangular_opt.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_triangular_opt.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_triangular_opt.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_triangular_opt.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_triangular_opt.layers.RandomSaturation(**(params["saturation"]))
                else:
                    self.optimize = False
                    params = {option: {} for option in parameters_name}
                    for option in parameters_name:
                        if domain_randomization.lowers is not None:
                            params[option]["lower"] = domain_randomization.lowers[parameters_name.index(option)]
                        if domain_randomization.modes is not None:
                            params[option]["mode"] = domain_randomization.modes[parameters_name.index(option)]
                        if domain_randomization.uppers is not None:
                            params[option]["upper"] = domain_randomization.uppers[parameters_name.index(option)]
                        if domain_randomization.factors is not None:
                            params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                        params[option]["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_triangular.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_triangular.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_triangular.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_triangular.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_triangular.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_triangular.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_triangular.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "univariate normal":
                if domain_randomization.optimize:
                    params = {}
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_univariatenormal_opt.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_univariatenormal_opt.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_univariatenormal_opt.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_univariatenormal_opt.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_univariatenormal_opt.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_univariatenormal_opt.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_univariatenormal_opt.layers.RandomSaturation(**(params["saturation"]))
                else:
                    self.optimize = False
                    params = {option: {} for option in parameters_name}
                    for option in parameters_name:
                        if domain_randomization.means is not None:
                            params[option]["mean"] = domain_randomization.means[parameters_name.index(option)]
                        if domain_randomization.variances is not None:
                            params[option]["variance"] = domain_randomization.variances[parameters_name.index(option)]
                        if domain_randomization.factors is not None:
                            params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                        params[option]["seed"] = domain_randomization.seed
                    self.branch1_random_brightness = r_univariatenormal.layers.RandomBrightness(**(params["brightness"]))
                    self.branch1_random_contrast = r_univariatenormal.layers.RandomContrast(**(params["contrast"]))
                    self.branch1_random_horizontally_flip = r_univariatenormal.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                    self.branch1_random_vertically_flip = r_univariatenormal.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                    self.branch1_random_hue = r_univariatenormal.layers.RandomHue(**(params["hue"]))
                    self.branch1_random_jpeg_quality = r_univariatenormal.layers.RandomJpegQuality(**(params["jpeg quality"]))
                    self.branch1_random_saturation = r_univariatenormal.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "multivariate normal":
                params = {}
                if domain_randomization.optimize:
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_parameters = r_multivariatenormal_opt.layers.RandomParameters(**params)
                else:
                    self.optimize = False
                    params["mean_vector"] = domain_randomization.mean_vector
                    params["variancecovariance_matrix"] = domain_randomization.variancecovariance_matrix
                    params["factors"] = domain_randomization.factors
                    params["seed"] = domain_randomization.seed
                    self.branch1_random_parameters = r_multivariatenormal.layers.RandomParameters(**params)
        else:
            self.domain_randomization = False
        self.domain_randomization__mode = domain_randomization.mode

        self.field = field

        if self.optimize:
            print("-> 2 optimizers needed when calling 'model.compile'\n")

        print("'model.compile' parameters info:")
        print("\tclass name required for loss and optimizers (e.g.: "
              "keras.losses.categorical_crossentropy or "
              "keras.optimizers.Adam or "
              "ng.optimizers.CMA")
        print("\tany additional parameters required must be passed as a dictionary in the second element of the tuple "
              "(class_name, parameters) (e.g.: (keras.optimizers.Adam, {'learning_rate':1e-3})")
        print("\t'run_eagerly=True' suggested.")


    def call(self, inputs, training=False):
        try:
            data = inputs[self.field]
        except KeyError:
            data = inputs
        if self.domain_randomization:
            if self.optimize and training:
                try: # self.domain_randomization__mode == "multivariate normal"
                    dr = self.branch1_random_parameters
                    sampled_params = tfp.distributions.MultivariateNormalTriL(
                        loc=self.branch2_mean_vector,
                        scale_tril=tfp.math.fill_triangular(
                            self.branch2_variancecovariance_matrix
                        )
                    ).sample(sample_shape=(data.shape[0],)).numpy()
                    data = dr(data, values=sampled_params, rand=False)
                except AttributeError:
                    if self.domain_randomization__mode == "uniform":
                        sampled_paramA = tfp.distributions.Uniform(
                            low=self.branch2_lowerA,
                            high=self.branch2_upperA
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramB = tfp.distributions.Uniform(
                            low=self.branch2_lowerB,
                            high=self.branch2_upperB
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramC = tfp.distributions.Uniform(
                            low=self.branch2_lowerC,
                            high=self.branch2_upperC
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramD = tfp.distributions.Uniform(
                            low=self.branch2_lowerD,
                            high=self.branch2_upperD
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramE = tfp.distributions.Uniform(
                            low=self.branch2_lowerE,
                            high=self.branch2_upperE
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramF = tfp.distributions.Uniform(
                            low=self.branch2_lowerF,
                            high=self.branch2_upperF
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramG = tfp.distributions.Uniform(
                            low=self.branch2_lowerG,
                            high=self.branch2_upperG
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                    if self.domain_randomization__mode == "triangular":
                        sampled_paramA = Triangular(
                            low=self.branch2_lowerA,
                            mode=self.branch2_modeA,
                            high=self.branch2_upperA
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramB = Triangular(
                            low=self.branch2_lowerB,
                            mode=self.branch2_modeB,
                            high=self.branch2_upperB
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramC = Triangular(
                            low=self.branch2_lowerC,
                            mode=self.branch2_modeC,
                            high=self.branch2_upperC
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramD = Triangular(
                            low=self.branch2_lowerD,
                            mode=self.branch2_modeD,
                            high=self.branch2_upperD
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramE = Triangular(
                            low=self.branch2_lowerE,
                            mode=self.branch2_modeE,
                            high=self.branch2_upperE
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramF = Triangular(
                            low=self.branch2_lowerF,
                            mode=self.branch2_modeF,
                            high=self.branch2_upperF
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramG = Triangular(
                            low=self.branch2_lowerG,
                            mode=self.branch2_modeG,
                            high=self.branch2_upperG
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                    if self.domain_randomization__mode == "univariate normal":
                        sampled_paramA = tfp.distributions.Normal(
                            loc=self.branch2_meanA,
                            scale=self.branch2_varianceA
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramB = tfp.distributions.Normal(
                            loc=self.branch2_meanB,
                            scale=self.branch2_varianceB
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramC = tfp.distributions.Normal(
                            loc=self.branch2_meanC,
                            scale=self.branch2_varianceC
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramD = tfp.distributions.Normal(
                            loc=self.branch2_meanD,
                            scale=self.branch2_varianceD
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramE = tfp.distributions.Normal(
                            loc=self.branch2_meanE,
                            scale=self.branch2_varianceE
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramF = tfp.distributions.Normal(
                            loc=self.branch2_meanF,
                            scale=self.branch2_varianceF
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                        sampled_paramG = tfp.distributions.Normal(
                            loc=self.branch2_meanG,
                            scale=self.branch2_varianceG
                        ).sample(sample_shape=(data.shape[0],)).numpy()
                    data = self.branch1_random_brightness(data, value=sampled_paramA, rand=False)
                    data = self.branch1_random_contrast(data, value=sampled_paramB, rand=False)
                    data = self.branch1_random_horizontally_flip(data, value=sampled_paramC, rand=False)
                    data = self.branch1_random_vertically_flip(data, value=sampled_paramD, rand=False)
                    data = self.branch1_random_hue(data, value=sampled_paramE, rand=False)
                    data = self.branch1_random_jpeg_quality(data, value=sampled_paramF, rand=False)
                    data = self.branch1_random_saturation(data, value=sampled_paramG, rand=False)
            else:
                try:
                    dr = self.branch1_random_parameters
                    data = dr(data)
                except AttributeError:
                    data = self.branch1_random_brightness(data)
                    data = self.branch1_random_contrast(data)
                    data = self.branch1_random_horizontally_flip(data)
                    data = self.branch1_random_vertically_flip(data)
                    data = self.branch1_random_hue(data)
                    data = self.branch1_random_jpeg_quality(data)
                    data = self.branch1_random_saturation(data)
        base_model_output = self.branch1_base_model(data)
        x = self.branch1_flatten(base_model_output)
        x = self.branch1_dense1(x)
        y = self.branch1_dense2(x)
        return y

    def train_step(self, data):
        if self.model_branch1.optimizer is None:
            par = None
            try:
                if isinstance(self.optimizer[0], tuple):
                    opt, par = self.optimizer[0]
                else:
                    opt = self.optimizer[0]
            except TypeError:
                if isinstance(self.optimizer, tuple):
                    opt, par = self.optimizer
                else:
                    opt = self.optimizer
            if par is not None:
                self.model_branch1.optimizer = opt(**par)
            else:
                self.model_branch1.optimizer = opt()
            self.__update_parameters(self.model_branch1.optimizer)
        if self.domain_randomization and self.optimize and self.model_branch2.optimizer is None:
            par = None
            if isinstance(self.optimizer[1], tuple):
                opt, par = self.optimizer[1]
            else:
                opt = self.optimizer[1]
            if par is None:
                par = {}
            trainableVars = [getattr(self, attr) for attr in self.model_branch2.get_trainable_variables()]
            if self.domain_randomization__mode != "multivariate normal":
                params = ng.p.Tuple(*trainableVars)
            else:
                params = ng.p.Tuple(*(list(chain(*trainableVars))))
            instrumentation = ng.p.Instrumentation(params=params)
            par["parametrization"] = instrumentation
            self.model_branch2.optimizer = opt(**par)
            self.__update_parameters(self.model_branch2.optimizer)
        if self.model_branch1.optimizer is None or ((self.domain_randomization and self.optimize) and self.model_branch2.optimizer is None):
            raise NotFoundOptimizerException()
        imgs, labs = data
        with tf.GradientTape(persistent=True) as tape:
            predictions = self(imgs, training=True)
            loss = self.compiled_loss(labs, predictions)
        loss_branch1 = loss
        if self.domain_randomization and self.optimize:
            if self.domain_randomization__mode == "uniform":
                # constraints: lowers<=uppers
                p = [[self.branch2_lowerA, self.branch2_upperA],
                     [self.branch2_lowerB, self.branch2_upperB],
                     [self.branch2_lowerC, self.branch2_upperC],
                     [self.branch2_lowerD, self.branch2_upperD],
                     [self.branch2_lowerE, self.branch2_upperE],
                     [self.branch2_lowerF, self.branch2_upperF],
                     [self.branch2_lowerG, self.branch2_upperG]
                     ]
                epsilon = 0.001
                ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.lowers__ranges, self.uppers__ranges)])
                width = ranges[:,1]-ranges[:,0]
                width = tf.where(tf.math.is_inf(width), 1000.0, width)
                rescaling_factor = width/(tf.reduce_max(width))
                penalty = tf.reduce_mean([ tf.maximum(tf.constant(0, dtype=tf.float32),
                                                      pi[0]-pi[1]+epsilon)
                                           /rescaling_factor[i]
                                           for i, pi in enumerate(p) ])
                loss_branch2 = loss+penalty
            if self.domain_randomization__mode == "triangular":
                # constraints: lowers<=modes, modes<=uppers
                p = [[self.branch2_lowerA, self.branch2_modeA, self.branch2_upperA],
                     [self.branch2_lowerB, self.branch2_modeB, self.branch2_upperA],
                     [self.branch2_lowerC, self.branch2_modeC, self.branch2_upperA],
                     [self.branch2_lowerD, self.branch2_modeD, self.branch2_upperA],
                     [self.branch2_lowerE, self.branch2_modeE, self.branch2_upperA],
                     [self.branch2_lowerF, self.branch2_modeF, self.branch2_upperA],
                     [self.branch2_lowerG, self.branch2_modeG, self.branch2_upperA]
                     ]
                epsilon = 0.001
                ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.lowers__ranges, self.modes__ranges)])
                width = ranges[:,1]-ranges[:,0]
                width = tf.where(tf.math.is_inf(width), 1000.0, width)
                rescaling_factor = width/(tf.reduce_max(width))
                penalty = tf.reduce_mean([ tf.maximum(tf.constant(0, dtype=tf.float32),
                                                      pi[0]-pi[1]+epsilon)
                                           /rescaling_factor[i]
                                           for i, pi in enumerate(p) ])
                loss_branch2 = loss+penalty
                ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.modes__ranges, self.uppers__ranges)])
                width = ranges[:,1]-ranges[:,0]
                width = tf.where(tf.math.is_inf(width), 1000.0, width)
                rescaling_factor = width/(tf.reduce_max(width))
                penalty = tf.reduce_mean([ tf.maximum(tf.constant(0, dtype=tf.float32),
                                                      pi[1]-pi[2]+epsilon)
                                           /rescaling_factor[i]
                                           for i, pi in enumerate(p) ])
                loss_branch2 += penalty
        self.__update_parameters(self.model_branch1.optimizer, loss_branch1, tape)
        if self.domain_randomization and self.optimize:
            self.__update_parameters(self.model_branch2.optimizer, loss_branch2)
        del tape

        self.compiled_metrics.update_state(labs, predictions)
        return {'loss': loss, **{m.name: m.result() for m in self.metrics}}


    def __update_parameters(self, optimizer, loss=None, tape=None):
        if is_nevergrad_optimizer(optimizer):
            if loss is not None:
                optimizer.tell(self.__tmp_optimizer_branch2_x, loss)
            x = optimizer.ask()
            parameters = list(x.kwargs['params'])
            if self.domain_randomization__mode == "uniform":
                for i in range(len(self.lowers__initials)):
                    p = parameters[i]
                    p = denormalize_value(
                        p,
                        self.lowers__ranges[i][0],
                        self.lowers__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i] = p
                for i in range(len(self.uppers__initials)):
                    p = parameters[i+len(self.lowers__initials)]
                    p = denormalize_value(
                        p,
                        self.uppers__ranges[i][0],
                        self.uppers__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i+len(self.lowers__initials)] = p
            elif self.domain_randomization__mode == "triangular":
                for i in range(len(self.lowers__initials)):
                    p = parameters[i]
                    p = denormalize_value(
                        p,
                        self.lowers__ranges[i][0],
                        self.lowers__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i] = p
                for i in range(len(self.modes__initials)):
                    p = parameters[i+len(self.lowers__initials)]
                    p = denormalize_value(
                        p,
                        self.modes__ranges[i][0],
                        self.modes__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i+len(self.lowers__initials)] = p
                for i in range(len(self.uppers__initials)):
                    p = parameters[i+len(self.modes__initials)]
                    p = denormalize_value(
                        p,
                        self.uppers__ranges[i][0],
                        self.uppers__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i+len(self.modes__initials)] = p
            elif self.domain_randomization__mode == "multivariate normal":
                for i in range(len(self.mean_vector__initials)):
                    p = parameters[i]
                    p = denormalize_value(
                        p,
                        self.mean_vector__ranges[i][0],
                        self.mean_vector__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i] = p
                for i in spiral_flat_from_progressive(
                        len(self.variancecovariance_matrix__initials)
                ): # mapping of indexes from progressive (flat) representation to spiral flat representation
                    p = parameters[i+len(self.mean_vector__initials)]
                    p = denormalize_value(
                        p,
                        self.variancecovariance_matrix__ranges[i][0],
                        self.variancecovariance_matrix__ranges[i][1],
                        self.normalized_space["lower"],
                        self.normalized_space["upper"]
                    )
                    parameters[i+len(self.mean_vector__initials)] = p
            if self.domain_randomization__mode != "multivariate normal":
                for var, new_var in zip(self.model_branch2.get_trainable_variables(), parameters):
                    setattr(self, var, new_var)
            else:
                mv = parameters[0:
                                len(self.mean_vector__initials)]
                vcv = parameters[len(self.mean_vector__initials):
                                 len(self.mean_vector__initials)+len(self.variancecovariance_matrix__initials)]
                for var, new_var in zip(self.model_branch2.get_trainable_variables(), [mv, vcv]):
                    setattr(self, var, new_var)
            self.__tmp_optimizer_branch2_x = x
        elif is_keras_optimizer(optimizer):
            if tape is not None:
                gradients = tape.gradient(loss, self.model_branch1.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model_branch1.trainable_variables))
        else:
            raise NotImplementedOptimizerException(optimizer)



class ResNet2__0__1(ResNet2__0):
    def __init__(self, n_classes, input_shape=(128, 128, 3), field='data'):
        super(ResNet2__0__1, self).__init__(n_classes=n_classes, input_shape=input_shape, field=field)
        # To be implemented:
        #       Even after two epochs, validation accuracy arrives near 90%. After 40 epochs the model
        #       comfortably converges. It is possible to reach up to higher accuracies by adding a couple of more fully
        #       connected layers. The successful results with only one hidden fully connected layer mean that ResNet-152
        #       does a pretty good job while extracting features for the classifier even though ImageNet and MNIST
        #       contain fairly distant image samples.

    def call(self, inputs):
        return self.model(inputs[self.field])




class ResNet2__1__1(ResNet2__0):
    def __init__(self, n_classes, input_shape=(128, 128, 3), field='data'):
        super(ResNet2__1__1, self).__init__(n_classes=n_classes, input_shape=input_shape, field=field)
        # To be implemented:
        #       Even after two epochs, validation accuracy arrives near 90%. After 40 epochs the model
        #       comfortably converges. It is possible to reach up to higher accuracies by adding a couple of more fully
        #       connected layers. The successful results with only one hidden fully connected layer mean that ResNet-152
        #       does a pretty good job while extracting features for the classifier even though ImageNet and MNIST
        #       contain fairly distant image samples.

    def call(self, inputs):
        return self.model(inputs[self.field])



class Branch():
    def __new__(cls, *args, name="default", **kwargs):
        if kwargs.get("inputs", None) is None or kwargs.get("outputs", None) is None:
            return Branch_a(name)
        else:
            kwargs["name"] = name
            return Branch_b(*args, **kwargs)

class Branch_a():
    def __init__(self, name="default"):
        self.__name = name
        self.__tr_vars = []

    def get_name(self):
            return self.__name

    def add_variables(self, var):
        if isinstance(var, list):
            for v in var:
                self.__tr_vars.append(v)
        else:
            self.__tr_vars.append(var)

    def get_trainable_variables(self):
        return self.__tr_vars

class Branch_b(keras.Model):
    def __init__(self, *args, **kwargs):
        super(Branch_b, self).__init__(*args, **kwargs)
        self.__name = kwargs["name"]

    def get_name(self):
        return self.__name