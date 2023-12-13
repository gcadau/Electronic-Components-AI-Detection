import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from keras import backend as K
import numpy as np
from algorithm.deep.utils import NotFoundOptimizerException, set_parameters__ranges, fill_matrix
from algorithm.domain_randomization.optimization.tf import (
    r_uniform as r_uniform_opt,
    r_triangular as r_triangular_opt,
    r_univariatenormal as r_univariatenormal_opt,
    r_multivariatenormal as r_multivariatenormal_opt
)
from algorithm.domain_randomization.tf import r_uniform, r_triangular, r_univariatenormal, r_multivariatenormal


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

            if domain_randomization.optimize:
                self.optimize = True

                self.branch2_flatten = keras.layers.Flatten()
                self.branch2_dense1 = keras.layers.Dense(128, activation='relu')
                self.branch2_dense2 = keras.layers.Dense(64, activation='relu')

                if domain_randomization.mode == "uniform":
                    if domain_randomization.lowers__ranges is None:
                        self.lowers__ranges = set_parameters__ranges("uniform", "lowers")
                    else:
                        self.lowers__ranges = domain_randomization.lowers__ranges
                    if domain_randomization.uppers__ranges is None:
                        self.uppers__ranges = set_parameters__ranges("uniform", "uppers")
                    else:
                        self.uppers__ranges = domain_randomization.uppers__ranges
                    self.branch2_dense3a = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.lowers__ranges))
                    self.branch2_dense3b = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.uppers__ranges))
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
                    self.branch2_dense3a = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.lowers__ranges))
                    self.branch2_dense3b = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.modes__ranges))
                    self.branch2_dense3c = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.uppers__ranges))
                elif domain_randomization.mode == "univariate normal":
                    if domain_randomization.means__ranges is None:
                        self.means__ranges = set_parameters__ranges("univariatenormal", "means")
                    else:
                        self.means__ranges = domain_randomization.means__ranges
                    if domain_randomization.variances__ranges is None:
                        self.variances__ranges = set_parameters__ranges("univariatenormal", "variances")
                    else:
                        self.variances__ranges = domain_randomization.variances__ranges
                    self.branch2_dense3a = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.means__ranges))
                    self.branch2_dense3b = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.variances__ranges))
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
                    self.branch2_dense3a = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.mean_vector__ranges))
                    self.branch2_dense3b = keras.layers.Dense(units=int(0.5*(len(domain_randomization.get_parameters_list())*(len(domain_randomization.get_parameters_list())-1)))+len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.variancecovariance_matrix__ranges))

                inputs = keras.layers.Input(shape=input_shape)
                x = self.branch2_flatten(inputs)
                x = self.branch2_dense1(x)
                x = self.branch2_dense2(x)
                self.branch2_y1 = self.branch2_dense3a(x)
                self.branch2_y2 = self.branch2_dense3b(x)
                if domain_randomization.mode == "triangular":
                    self.branch2_y3 = self.branch2_dense3c(x)

                if domain_randomization.mode != "triangular":
                    self.model_branch2 = Branch(inputs=inputs, outputs=[self.branch2_y1, self.branch2_y2], name="Random Distribution parameters branch")
                else:
                    self.model_branch2 = Branch(inputs=inputs, outputs=[self.branch2_y1, self.branch2_y2, self.branch2_y3], name="Random Distribution parameters branch")
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
                    self.domain_randomization.optimize = False
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
                    self.domain_randomization.optimize = False
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
                    self.domain_randomization.optimize = False
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
                    self.domain_randomization.optimize = False
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
            print("2 optimizers needed when calling 'model.compile'")


    def call(self, inputs):
        try:
            data = inputs[self.field]
        except KeyError:
            data = inputs
        if self.domain_randomization:
            if self.optimize and training:
                x = self.branch2_flatten(data)
                x = self.branch2_dense1(x)
                x = self.branch2_dense2(x)
                y1 = self.branch2_dense3a(x)
                y2 = self.branch2_dense3b(x)
                try:
                    y3 = self.branch2_dense3c(x)
                except AttributeError:
                    y3 = None
                try:
                    dr = self.branch1_random_parameters
                    mean_vector = np.array(y1)
                    variancecovariance_matrix = np.array(fill_matrix(np.array(y2)))
                    data = dr(data, mean_vector, variancecovariance_matrix)
                except AttributeError:
                    p1 = np.array(y1)
                    p2 = np.array(y2)
                    if y3 is not None:
                        p3 = np.array(y3)
                    else:
                        p3 = None
                    if p3 is None:
                        data = self.branch1_random_brightness(data, p1, p2)
                        data = self.branch1_random_contrast(data, p1, p2)
                        data = self.branch1_random_horizontally_flip(data, p1, p2)
                        data = self.branch1_random_vertically_flip(data, p1, p2)
                        data = self.branch1_random_hue(data, p1, p2)
                        data = self.branch1_random_jpeg_quality(data, p1, p2)
                        data = self.branch1_random_saturation(data, p1, p2)
                    else:
                        data = self.branch1_random_brightness(data, p1, p2, p3)
                        data = self.branch1_random_contrast(data, p1, p2, p3)
                        data = self.branch1_random_horizontally_flip(data, p1, p2, p3)
                        data = self.branch1_random_vertically_flip(data, p1, p2, p3)
                        data = self.branch1_random_hue(data, p1, p2, p3)
                        data = self.branch1_random_jpeg_quality(data, p1, p2, p3)
                        data = self.branch1_random_saturation(data, p1, p2, p3)
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
        x = self.conv_1(data)
        x = self.conv_2(x)
        x = self.maxpool(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.conv_3(x)
        x = self.glopool(x)
        x = self.dense_1(x)
        x = self.do(x)
        x = self.dense_2(x)
        return x

    def train_step(self, data):
        imgs, labs = data
        with tf.GradientTape(persistent=True) as tape:
            predictions = self(imgs, training=True)
            loss = self.compiled_loss(labs, predictions)
            if self.domain_randomization and self.optimize:
                if self.domain_randomization__mode == "uniform":
                    # constraints: lowers<=uppers
                    try:
                        data = imgs[self.field]
                    except KeyError:
                        data = imgs
                    p = self.model_branch2(data)
                    p1 = p[0]
                    p2 = p[1]
                    epsilon = 0.001
                    ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.lowers__ranges, self.uppers__ranges)])
                    width = ranges[:,1]-ranges[:,0]
                    width = tf.where(tf.math.is_inf(width), 1000.0, width)
                    rescaling_factor = width/(tf.reduce_max(width))
                    penalty = tf.reduce_mean(tf.reduce_mean(tf.maximum(tf.zeros(shape=(p1.shape[1]), dtype=tf.float32), p1 - p2 + epsilon),
                                                            axis=0)/rescaling_factor, axis=0)
                    loss = loss+penalty
                if self.domain_randomization__mode == "triangular":
                    # constraints: lowers<=modes, modes<=uppers
                    try:
                        data = imgs[self.field]
                    except KeyError:
                        data = imgs
                    p = self.model_branch2(data)
                    p1 = p[0]
                    p2 = p[1]
                    p3 = p[2]
                    epsilon = 0.001
                    ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.lowers__ranges, self.modes__ranges)])
                    width = ranges[:,1]-ranges[:,0]
                    width = tf.where(tf.math.is_inf(width), 1000.0, width)
                    rescaling_factor = width/(tf.reduce_max(width))
                    penalty = tf.reduce_mean(tf.reduce_mean(tf.maximum(tf.zeros(shape=(p1.shape[1]), dtype=tf.float32), p1 - p2 + epsilon),
                                                            axis=0)/rescaling_factor, axis=0)
                    loss += penalty
                    ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.modes__ranges, self.uppers__ranges)])
                    width = ranges[:,1]-ranges[:,0]
                    width = tf.where(tf.math.is_inf(width), 1000.0, width)
                    rescaling_factor = width/(tf.reduce_max(width))
                    penalty = tf.reduce_mean(tf.reduce_mean(tf.maximum(tf.zeros(shape=(p1.shape[1]), dtype=tf.float32), p2 - p3 + epsilon),
                                                            axis=0)/rescaling_factor, axis=0)
                    loss += penalty
        gradients_branch1 = tape.gradient(loss, self.model_branch1.trainable_variables)
        if self.model_branch1.optimizer is None:
            self.model_branch1.optimizer = self.optimizer[0]
        if self.domain_randomization and self.optimize:
            gradients_branch2 = tape.gradient(loss, self.model_branch2.trainable_variables)
            if self.model_branch2.optimizer is None:
                self.model_branch2.optimizer = self.optimizer[1]
        if self.model_branch1.optimizer is None or ((self.domain_randomization and self.optimize) and self.model_branch2.optimizer is None):
            raise NotFoundOptimizerException()
        self.model_branch1.optimizer.apply_gradients(zip(gradients_branch1, self.model_branch1.trainable_variables))
        if self.domain_randomization and self.optimize:
            self.model_branch2.optimizer.apply_gradients(zip(gradients_branch2, self.model_branch2.trainable_variables))
        del tape
        return {"loss": loss}




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

            if domain_randomization.optimize:
                self.optimize = True

                self.branch2_flatten = keras.layers.Flatten()
                self.branch2_dense1 = keras.layers.Dense(128, activation='relu')
                self.branch2_dense2 = keras.layers.Dense(64, activation='relu')

                if domain_randomization.mode == "uniform":
                    if domain_randomization.lowers__ranges is None:
                        self.lowers__ranges = set_parameters__ranges("uniform", "lowers")
                    else:
                        self.lowers__ranges = domain_randomization.lowers__ranges
                    if domain_randomization.uppers__ranges is None:
                        self.uppers__ranges = set_parameters__ranges("uniform", "uppers")
                    else:
                        self.uppers__ranges = domain_randomization.uppers__ranges
                    self.branch2_dense3a = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.lowers__ranges))
                    self.branch2_dense3b = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.uppers__ranges))
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
                    self.branch2_dense3a = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.lowers__ranges))
                    self.branch2_dense3b = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.modes__ranges))
                    self.branch2_dense3c = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.uppers__ranges))
                elif domain_randomization.mode == "univariate normal":
                    if domain_randomization.means__ranges is None:
                        self.means__ranges = set_parameters__ranges("univariatenormal", "means")
                    else:
                        self.means__ranges = domain_randomization.means__ranges
                    if domain_randomization.variances__ranges is None:
                        self.variances__ranges = set_parameters__ranges("univariatenormal", "variances")
                    else:
                        self.variances__ranges = domain_randomization.variances__ranges
                    self.branch2_dense3a = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.means__ranges))
                    self.branch2_dense3b = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.variances__ranges))
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
                    self.branch2_dense3a = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.mean_vector__ranges))
                    self.branch2_dense3b = keras.layers.Dense(units=int(0.5*(len(domain_randomization.get_parameters_list())*(len(domain_randomization.get_parameters_list())-1)))+len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.variancecovariance_matrix__ranges))

                inputs = keras.layers.Input(shape=input_shape)
                x = self.branch2_flatten(inputs)
                x = self.branch2_dense1(x)
                x = self.branch2_dense2(x)
                self.branch2_y1 = self.branch2_dense3a(x)
                self.branch2_y2 = self.branch2_dense3b(x)
                if domain_randomization.mode == "triangular":
                    self.branch2_y3 = self.branch2_dense3c(x)

                if domain_randomization.mode != "triangular":
                    self.model_branch2 = Branch(inputs=inputs, outputs=[self.branch2_y1, self.branch2_y2], name="Random Distribution parameters branch")
                else:
                    self.model_branch2 = Branch(inputs=inputs, outputs=[self.branch2_y1, self.branch2_y2, self.branch2_y3], name="Random Distribution parameters branch")
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
                    self.domain_randomization.optimize = False
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
                    self.domain_randomization.optimize = False
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
                    self.domain_randomization.optimize = False
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
                    self.domain_randomization.optimize = False
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
            print("2 optimizers needed when calling 'model.compile'")


    def call(self, inputs, training=False):
        try:
            data = inputs[self.field]
        except KeyError:
            data = inputs
        if self.domain_randomization:
            if self.optimize and training:
                x = self.branch2_flatten(data)
                x = self.branch2_dense1(x)
                x = self.branch2_dense2(x)
                y1 = self.branch2_dense3a(x)
                y2 = self.branch2_dense3b(x)
                try:
                    y3 = self.branch2_dense3c(x)
                except AttributeError:
                    y3 = None
                try:
                    dr = self.branch1_random_parameters
                    mean_vector = np.array(y1)
                    variancecovariance_matrix = np.array(fill_matrix(np.array(y2)))
                    data = dr(data, mean_vector, variancecovariance_matrix)
                except AttributeError:
                    p1 = np.array(y1)
                    p2 = np.array(y2)
                    if y3 is not None:
                        p3 = np.array(y3)
                    else:
                        p3 = None
                    if p3 is None:
                        data = self.branch1_random_brightness(data, p1, p2)
                        data = self.branch1_random_contrast(data, p1, p2)
                        data = self.branch1_random_horizontally_flip(data, p1, p2)
                        data = self.branch1_random_vertically_flip(data, p1, p2)
                        data = self.branch1_random_hue(data, p1, p2)
                        data = self.branch1_random_jpeg_quality(data, p1, p2)
                        data = self.branch1_random_saturation(data, p1, p2)
                    else:
                        data = self.branch1_random_brightness(data, p1, p2, p3)
                        data = self.branch1_random_contrast(data, p1, p2, p3)
                        data = self.branch1_random_horizontally_flip(data, p1, p2, p3)
                        data = self.branch1_random_vertically_flip(data, p1, p2, p3)
                        data = self.branch1_random_hue(data, p1, p2, p3)
                        data = self.branch1_random_jpeg_quality(data, p1, p2, p3)
                        data = self.branch1_random_saturation(data, p1, p2, p3)
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
        imgs, labs = data
        with tf.GradientTape(persistent=True) as tape:
            predictions = self(imgs, training=True)
            loss = self.compiled_loss(labs, predictions)
            if self.domain_randomization and self.optimize:
                if self.domain_randomization__mode == "uniform":
                    # constraints: lowers<=uppers
                    try:
                        data = imgs[self.field]
                    except KeyError:
                        data = imgs
                    p = self.model_branch2(data)
                    p1 = p[0]
                    p2 = p[1]
                    epsilon = 0.001
                    ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.lowers__ranges, self.uppers__ranges)])
                    width = ranges[:,1]-ranges[:,0]
                    width = tf.where(tf.math.is_inf(width), 1000.0, width)
                    rescaling_factor = width/(tf.reduce_max(width))
                    penalty = tf.reduce_mean(tf.reduce_mean(tf.maximum(tf.zeros(shape=(p1.shape[1]), dtype=tf.float32), p1 - p2 + epsilon),
                                             axis=0)/rescaling_factor, axis=0)
                    loss = loss+penalty
                if self.domain_randomization__mode == "triangular":
                    # constraints: lowers<=modes, modes<=uppers
                    try:
                        data = imgs[self.field]
                    except KeyError:
                        data = imgs
                    p = self.model_branch2(data)
                    p1 = p[0]
                    p2 = p[1]
                    p3 = p[2]
                    epsilon = 0.001
                    ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.lowers__ranges, self.modes__ranges)])
                    width = ranges[:,1]-ranges[:,0]
                    width = tf.where(tf.math.is_inf(width), 1000.0, width)
                    rescaling_factor = width/(tf.reduce_max(width))
                    penalty = tf.reduce_mean(tf.reduce_mean(tf.maximum(tf.zeros(shape=(p1.shape[1]), dtype=tf.float32), p1 - p2 + epsilon),
                                                            axis=0)/rescaling_factor, axis=0)
                    loss += penalty
                    ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.modes__ranges, self.uppers__ranges)])
                    width = ranges[:,1]-ranges[:,0]
                    width = tf.where(tf.math.is_inf(width), 1000.0, width)
                    rescaling_factor = width/(tf.reduce_max(width))
                    penalty = tf.reduce_mean(tf.reduce_mean(tf.maximum(tf.zeros(shape=(p1.shape[1]), dtype=tf.float32), p2 - p3 + epsilon),
                                                            axis=0)/rescaling_factor, axis=0)
                    loss += penalty
        gradients_branch1 = tape.gradient(loss, self.model_branch1.trainable_variables)
        if self.model_branch1.optimizer is None:
            self.model_branch1.optimizer = self.optimizer[0]
        if self.domain_randomization and self.optimize:
            gradients_branch2 = tape.gradient(loss, self.model_branch2.trainable_variables)
            if self.model_branch2.optimizer is None:
                self.model_branch2.optimizer = self.optimizer[1]
        if self.model_branch1.optimizer is None or ((self.domain_randomization and self.optimize) and self.model_branch2.optimizer is None):
            raise NotFoundOptimizerException()
        self.model_branch1.optimizer.apply_gradients(zip(gradients_branch1, self.model_branch1.trainable_variables))
        if self.domain_randomization and self.optimize:
            self.model_branch2.optimizer.apply_gradients(zip(gradients_branch2, self.model_branch2.trainable_variables))
        del tape
        return {"loss": loss}




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

            if domain_randomization.optimize:
                self.optimize = True

                self.branch2_flatten = keras.layers.Flatten()
                self.branch2_dense1 = keras.layers.Dense(128, activation='relu')
                self.branch2_dense2 = keras.layers.Dense(64, activation='relu')

                if domain_randomization.mode == "uniform":
                    if domain_randomization.lowers__ranges is None:
                        self.lowers__ranges = set_parameters__ranges("uniform", "lowers")
                    else:
                        self.lowers__ranges = domain_randomization.lowers__ranges
                    if domain_randomization.uppers__ranges is None:
                        self.uppers__ranges = set_parameters__ranges("uniform", "uppers")
                    else:
                        self.uppers__ranges = domain_randomization.uppers__ranges
                    self.branch2_dense3a = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.lowers__ranges))
                    self.branch2_dense3b = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.uppers__ranges))
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
                    self.branch2_dense3a = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.lowers__ranges))
                    self.branch2_dense3b = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.modes__ranges))
                    self.branch2_dense3c = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.uppers__ranges))
                elif domain_randomization.mode == "univariate normal":
                    if domain_randomization.means__ranges is None:
                        self.means__ranges = set_parameters__ranges("univariatenormal", "means")
                    else:
                        self.means__ranges = domain_randomization.means__ranges
                    if domain_randomization.variances__ranges is None:
                        self.variances__ranges = set_parameters__ranges("univariatenormal", "variances")
                    else:
                        self.variances__ranges = domain_randomization.variances__ranges
                    self.branch2_dense3a = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.means__ranges))
                    self.branch2_dense3b = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.variances__ranges))
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
                    self.branch2_dense3a = keras.layers.Dense(units=len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.mean_vector__ranges))
                    self.branch2_dense3b = keras.layers.Dense(units=int(0.5*(len(domain_randomization.get_parameters_list())*(len(domain_randomization.get_parameters_list())-1)))+len(domain_randomization.get_parameters_list()),
                                                              activation=lambda x: sigmoid_activation__withConstraints(x, self.variancecovariance_matrix__ranges))

                inputs = keras.layers.Input(shape=input_shape)
                x = self.branch2_flatten(inputs)
                x = self.branch2_dense1(x)
                x = self.branch2_dense2(x)
                self.branch2_y1 = self.branch2_dense3a(x)
                self.branch2_y2 = self.branch2_dense3b(x)
                if domain_randomization.mode == "triangular":
                    self.branch2_y3 = self.branch2_dense3c(x)

                if domain_randomization.mode != "triangular":
                    self.model_branch2 = Branch(inputs=inputs, outputs=[self.branch2_y1, self.branch2_y2], name="Random Distribution parameters branch")
                else:
                    self.model_branch2 = Branch(inputs=inputs, outputs=[self.branch2_y1, self.branch2_y2, self.branch2_y3], name="Random Distribution parameters branch")
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
                    self.domain_randomization.optimize = False
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
                    self.domain_randomization.optimize = False
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
                    self.domain_randomization.optimize = False
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
                    self.domain_randomization.optimize = False
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
            print("2 optimizers needed when calling 'model.compile'")


    def call(self, inputs, training=False):
        try:
            data = inputs[self.field]
        except KeyError:
            data = inputs
        if self.domain_randomization:
            if self.optimize and training:
                x = self.branch2_flatten(data)
                x = self.branch2_dense1(x)
                x = self.branch2_dense2(x)
                y1 = self.branch2_dense3a(x)
                y2 = self.branch2_dense3b(x)
                try:
                    y3 = self.branch2_dense3c(x)
                except AttributeError:
                    y3 = None
                try:
                    dr = self.branch1_random_parameters
                    mean_vector = np.array(y1)
                    variancecovariance_matrix = np.array(fill_matrix(np.array(y2)))
                    data = dr(data, mean_vector, variancecovariance_matrix)
                except AttributeError:
                    p1 = np.array(y1)
                    p2 = np.array(y2)
                    if y3 is not None:
                        p3 = np.array(y3)
                    else:
                        p3 = None
                    if p3 is None:
                        data = self.branch1_random_brightness(data, p1, p2)
                        data = self.branch1_random_contrast(data, p1, p2)
                        data = self.branch1_random_horizontally_flip(data, p1, p2)
                        data = self.branch1_random_vertically_flip(data, p1, p2)
                        data = self.branch1_random_hue(data, p1, p2)
                        data = self.branch1_random_jpeg_quality(data, p1, p2)
                        data = self.branch1_random_saturation(data, p1, p2)
                    else:
                        data = self.branch1_random_brightness(data, p1, p2, p3)
                        data = self.branch1_random_contrast(data, p1, p2, p3)
                        data = self.branch1_random_horizontally_flip(data, p1, p2, p3)
                        data = self.branch1_random_vertically_flip(data, p1, p2, p3)
                        data = self.branch1_random_hue(data, p1, p2, p3)
                        data = self.branch1_random_jpeg_quality(data, p1, p2, p3)
                        data = self.branch1_random_saturation(data, p1, p2, p3)
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
        imgs, labs = data
        with tf.GradientTape(persistent=True) as tape:
            predictions = self(imgs, training=True)
            loss = self.compiled_loss(labs, predictions)
            if self.domain_randomization and self.optimize:
                if self.domain_randomization__mode == "uniform":
                    # constraints: lowers<=uppers
                    try:
                        data = imgs[self.field]
                    except KeyError:
                        data = imgs
                    p = self.model_branch2(data)
                    p1 = p[0]
                    p2 = p[1]
                    epsilon = 0.001
                    ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.lowers__ranges, self.uppers__ranges)])
                    width = ranges[:,1]-ranges[:,0]
                    width = tf.where(tf.math.is_inf(width), 1000.0, width)
                    rescaling_factor = width/(tf.reduce_max(width))
                    penalty = tf.reduce_mean(tf.reduce_mean(tf.maximum(tf.zeros(shape=(p1.shape[1]), dtype=tf.float32), p1 - p2 + epsilon),
                                                            axis=0)/rescaling_factor, axis=0)
                    loss = loss+penalty
                if self.domain_randomization__mode == "triangular":
                    # constraints: lowers<=modes, modes<=uppers
                    try:
                        data = imgs[self.field]
                    except KeyError:
                        data = imgs
                    p = self.model_branch2(data)
                    p1 = p[0]
                    p2 = p[1]
                    p3 = p[2]
                    epsilon = 0.001
                    ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.lowers__ranges, self.modes__ranges)])
                    width = ranges[:,1]-ranges[:,0]
                    width = tf.where(tf.math.is_inf(width), 1000.0, width)
                    rescaling_factor = width/(tf.reduce_max(width))
                    penalty = tf.reduce_mean(tf.reduce_mean(tf.maximum(tf.zeros(shape=(p1.shape[1]), dtype=tf.float32), p1 - p2 + epsilon),
                                                            axis=0)/rescaling_factor, axis=0)
                    loss += penalty
                    ranges = tf.constant([[a[0], b[1]] for a, b in zip(self.modes__ranges, self.uppers__ranges)])
                    width = ranges[:,1]-ranges[:,0]
                    width = tf.where(tf.math.is_inf(width), 1000.0, width)
                    rescaling_factor = width/(tf.reduce_max(width))
                    penalty = tf.reduce_mean(tf.reduce_mean(tf.maximum(tf.zeros(shape=(p1.shape[1]), dtype=tf.float32), p2 - p3 + epsilon),
                                                            axis=0)/rescaling_factor, axis=0)
                    loss += penalty
        gradients_branch1 = tape.gradient(loss, self.model_branch1.trainable_variables)
        if self.model_branch1.optimizer is None:
            self.model_branch1.optimizer = self.optimizer[0]
        if self.domain_randomization and self.optimize:
            gradients_branch2 = tape.gradient(loss, self.model_branch2.trainable_variables)
            if self.model_branch2.optimizer is None:
                self.model_branch2.optimizer = self.optimizer[1]
        if self.model_branch1.optimizer is None or ((self.domain_randomization and self.optimize) and self.model_branch2.optimizer is None):
            raise NotFoundOptimizerException()
        self.model_branch1.optimizer.apply_gradients(zip(gradients_branch1, self.model_branch1.trainable_variables))
        if self.domain_randomization and self.optimize:
            self.model_branch2.optimizer.apply_gradients(zip(gradients_branch2, self.model_branch2.trainable_variables))
            del tape
        return {"loss": loss}



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



class Branch(keras.Model):
    def __init__(self, *args, name="default", **kwargs):
        super(Branch, self).__init__(*args, **kwargs)
        self.__name = name

    def get_name(self):
        return self.__name






def sigmoid_activation__withConstraints(x, output_constraints):
    constrained_outputs = []
    for i, l in enumerate(output_constraints):
        min_val, max_val = tuple(l)
        constrained_output = (max_val - min_val) * K.sigmoid(x[:, i]) + min_val
        constrained_outputs.append(constrained_output)
    return K.stack(constrained_outputs, axis=-1)