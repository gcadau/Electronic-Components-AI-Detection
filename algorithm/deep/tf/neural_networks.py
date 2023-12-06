from tensorflow import keras
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
        self.conv_1 = keras.layers.Conv2D(32, 3, activation="relu", input_shape=input_shape)
        self.conv_2 = keras.layers.Conv2D(64, 3, activation="relu")
        self.maxpool = keras.layers.MaxPooling2D(3)
        self.block_1 = ResNetBlock((64, 64))
        self.block_2 = ResNetBlock((64, 64))
        self.conv_3 = keras.layers.Conv2D(64, 3, activation='relu')
        self.glopool = keras.layers.GlobalAveragePooling2D()
        self.dense_1 = keras.layers.Dense(256, activation="relu")
        self.do = keras.layers.Dropout(0.5)
        self.dense_2 = keras.layers.Dense(n_classes)

        if domain_randomization is not None:
            self.domain_randomization = True
            parameters_name = domain_randomization.get_parameters_list()
            if domain_randomization.mode == "uniform":
                params = {option: {} for option in parameters_name}
                for option in parameters_name:
                    if domain_randomization.lowers is not None:
                        params[option]["lower"] = domain_randomization.lowers[parameters_name.index(option)]
                    if domain_randomization.uppers is not None:
                        params[option]["upper"] = domain_randomization.uppers[parameters_name.index(option)]
                    if domain_randomization.factors is not None:
                        params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                self.random_brightness = r_uniform.layers.RandomBrightness(**(params["brightness"]))
                self.random_contrast = r_uniform.layers.RandomContrast(**(params["contrast"]))
                self.random_horizontally_flip = r_uniform.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                self.random_vertically_flip = r_uniform.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                self.random_hue = r_uniform.layers.RandomHue(**(params["hue"]))
                self.random_jpeg_quality = r_uniform.layers.RandomJpegQuality(**(params["jpeg quality"]))
                self.random_saturation = r_uniform.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "triangular":
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
                self.random_brightness = r_triangular.layers.RandomBrightness(**(params["brightness"]))
                self.random_contrast = r_triangular.layers.RandomContrast(**(params["contrast"]))
                self.random_horizontally_flip = r_triangular.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                self.random_vertically_flip = r_triangular.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                self.random_hue = r_triangular.layers.RandomHue(**(params["hue"]))
                self.random_jpeg_quality = r_triangular.layers.RandomJpegQuality(**(params["jpeg quality"]))
                self.random_saturation = r_triangular.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "univariate normal":
                params = {option: {} for option in parameters_name}
                for option in parameters_name:
                    if domain_randomization.means is not None:
                        params[option]["mean"] = domain_randomization.means[parameters_name.index(option)]
                    if domain_randomization.variances is not None:
                        params[option]["variance"] = domain_randomization.variances[parameters_name.index(option)]
                    if domain_randomization.factors is not None:
                        params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                self.random_brightness = r_univariatenormal.layers.RandomBrightness(**(params["brightness"]))
                self.random_contrast = r_univariatenormal.layers.RandomContrast(**(params["contrast"]))
                self.random_horizontally_flip = r_univariatenormal.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                self.random_vertically_flip = r_univariatenormal.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                self.random_hue = r_univariatenormal.layers.RandomHue(**(params["hue"]))
                self.random_jpeg_quality = r_univariatenormal.layers.RandomJpegQuality(**(params["jpeg quality"]))
                self.random_saturation = r_univariatenormal.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "multivariate normal":
                params = {}
                params["mean_vector"] = domain_randomization.mean_vector
                params["variancecovariance_matrix"] = domain_randomization.variancecovariance_matrix
                params["factor"] = domain_randomization.factors
                self.random_parameters = r_multivariatenormal.layers.RandomParameters(**params)
        else:
            self.domain_randomization = False

        self.field = field

    def call(self, inputs):
        data = inputs[self.field]
        if self.domain_randomization:
            try:
                dr = self.random_parameters
                data = dr(data)
            except AttributeError:
                data = self.random_brightness(data)
                data = self.random_contrast(data)
                data = self.random_horizontally_flip(data)
                data = self.random_vertically_flip(data)
                data = self.random_hue(data)
                data = self.random_jpeg_quality(data)
                data = self.random_saturation(data)
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




class ResNet2__0(keras.Model):
    def __init__(self, n_classes, input_shape=(128, 128, 3), field='data', domain_randomization=None):
        super(ResNet2__0, self).__init__()
        self.base_model = keras.applications.ResNet152(weights = 'imagenet', include_top = False, input_shape = input_shape)
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(1000, activation='relu')
        self.dense2 = keras.layers.Dense(n_classes, activation='softmax')

        x = self.flatten(self.base_model.output)
        x = self.dense1(x)
        self.y = self.dense2(x)

        self.model = keras.Model(inputs=self.base_model.input, outputs=self.y)

        if domain_randomization is not None:
            self.domain_randomization = True
            parameters_name = domain_randomization.get_parameters_list()
            if domain_randomization.mode == "uniform":
                params = {option: {} for option in parameters_name}
                for option in parameters_name:
                    if domain_randomization.lowers is not None:
                        params[option]["lower"] = domain_randomization.lowers[parameters_name.index(option)]
                    if domain_randomization.uppers is not None:
                        params[option]["upper"] = domain_randomization.uppers[parameters_name.index(option)]
                    if domain_randomization.factors is not None:
                        params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                    params[option]["seed"] = domain_randomization.seed
                self.random_brightness = r_uniform.layers.RandomBrightness(**(params["brightness"]))
                self.random_contrast = r_uniform.layers.RandomContrast(**(params["contrast"]))
                self.random_horizontally_flip = r_uniform.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                self.random_vertically_flip = r_uniform.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                self.random_hue = r_uniform.layers.RandomHue(**(params["hue"]))
                self.random_jpeg_quality = r_uniform.layers.RandomJpegQuality(**(params["jpeg quality"]))
                self.random_saturation = r_uniform.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "triangular":
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
                self.random_brightness = r_triangular.layers.RandomBrightness(**(params["brightness"]))
                self.random_contrast = r_triangular.layers.RandomContrast(**(params["contrast"]))
                self.random_horizontally_flip = r_triangular.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                self.random_vertically_flip = r_triangular.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                self.random_hue = r_triangular.layers.RandomHue(**(params["hue"]))
                self.random_jpeg_quality = r_triangular.layers.RandomJpegQuality(**(params["jpeg quality"]))
                self.random_saturation = r_triangular.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "univariate normal":
                params = {option: {} for option in parameters_name}
                for option in parameters_name:
                    if domain_randomization.means is not None:
                        params[option]["mean"] = domain_randomization.means[parameters_name.index(option)]
                    if domain_randomization.variances is not None:
                        params[option]["variance"] = domain_randomization.variances[parameters_name.index(option)]
                    if domain_randomization.factors is not None:
                        params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                    params[option]["seed"] = domain_randomization.seed
                self.random_brightness = r_univariatenormal.layers.RandomBrightness(**(params["brightness"]))
                self.random_contrast = r_univariatenormal.layers.RandomContrast(**(params["contrast"]))
                self.random_horizontally_flip = r_univariatenormal.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                self.random_vertically_flip = r_univariatenormal.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                self.random_hue = r_univariatenormal.layers.RandomHue(**(params["hue"]))
                self.random_jpeg_quality = r_univariatenormal.layers.RandomJpegQuality(**(params["jpeg quality"]))
                self.random_saturation = r_univariatenormal.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "multivariate normal":
                params = {}
                params["mean_vector"] = domain_randomization.mean_vector
                params["variancecovariance_matrix"] = domain_randomization.variancecovariance_matrix
                params["factors"] = domain_randomization.factors
                params["seed"] = domain_randomization.seed
                self.random_parameters = r_multivariatenormal.layers.RandomParameters(**params)
        else:
            self.domain_randomization = False

        self.field = field


    def call(self, inputs):
        try:
            data = inputs[self.field]
        except KeyError:
            data = inputs
        if self.domain_randomization:
            try:
                dr = self.random_parameters
                data = dr(data)
            except AttributeError:
                data = self.random_brightness(data)
                data = self.random_contrast(data)
                data = self.random_horizontally_flip(data)
                data = self.random_vertically_flip(data)
                data = self.random_hue(data)
                data = self.random_jpeg_quality(data)
                data = self.random_saturation(data)
        base_model_output = self.base_model(data)
        x = self.flatten(base_model_output)
        x = self.dense1(x)
        y = self.dense2(x)
        return y




class ResNet2__1(keras.Model):
    def __init__(self, n_classes, input_shape=(128, 128, 3), field='data', domain_randomization=None):
        super(ResNet2__1, self).__init__()
        self.base_model = keras.applications.ResNet152(weights = 'imagenet', include_top = False, input_shape = input_shape)
        for layer in self.base_model.layers:
            layer.trainable = False
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(1000, activation='relu')
        self.dense2 = keras.layers.Dense(n_classes, activation='softmax')

        x = self.flatten(self.base_model.output)
        x = self.dense1(x)
        self.y = self.dense2(x)

        self.model = keras.Model(inputs=self.base_model.input, outputs=self.y)

        if domain_randomization is not None:
            self.domain_randomization = True
            parameters_name = domain_randomization.get_parameters_list()
            if domain_randomization.mode == "uniform":
                params = {option: {} for option in parameters_name}
                for option in parameters_name:
                    if domain_randomization.lowers is not None:
                        params[option]["lower"] = domain_randomization.lowers[parameters_name.index(option)]
                    if domain_randomization.uppers is not None:
                        params[option]["upper"] = domain_randomization.uppers[parameters_name.index(option)]
                    if domain_randomization.factors is not None:
                        params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                    params[option]["seed"] = domain_randomization.seed
                self.random_brightness = r_uniform.layers.RandomBrightness(**(params["brightness"]))
                self.random_contrast = r_uniform.layers.RandomContrast(**(params["contrast"]))
                self.random_horizontally_flip = r_uniform.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                self.random_vertically_flip = r_uniform.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                self.random_hue = r_uniform.layers.RandomHue(**(params["hue"]))
                self.random_jpeg_quality = r_uniform.layers.RandomJpegQuality(**(params["jpeg quality"]))
                self.random_saturation = r_uniform.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "triangular":
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
                self.random_brightness = r_triangular.layers.RandomBrightness(**(params["brightness"]))
                self.random_contrast = r_triangular.layers.RandomContrast(**(params["contrast"]))
                self.random_horizontally_flip = r_triangular.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                self.random_vertically_flip = r_triangular.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                self.random_hue = r_triangular.layers.RandomHue(**(params["hue"]))
                self.random_jpeg_quality = r_triangular.layers.RandomJpegQuality(**(params["jpeg quality"]))
                self.random_saturation = r_triangular.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "univariate normal":
                params = {option: {} for option in parameters_name}
                for option in parameters_name:
                    if domain_randomization.means is not None:
                        params[option]["mean"] = domain_randomization.means[parameters_name.index(option)]
                    if domain_randomization.variances is not None:
                        params[option]["variance"] = domain_randomization.variances[parameters_name.index(option)]
                    if domain_randomization.factors is not None:
                        params[option]["factor"] = domain_randomization.factors[parameters_name.index(option)]
                    params[option]["seed"] = domain_randomization.seed
                self.random_brightness = r_univariatenormal.layers.RandomBrightness(**(params["brightness"]))
                self.random_contrast = r_univariatenormal.layers.RandomContrast(**(params["contrast"]))
                self.random_horizontally_flip = r_univariatenormal.layers.RandomHorizontallyFlip(**(params["horizontally flip"]))
                self.random_vertically_flip = r_univariatenormal.layers.RandomVerticallyFlip(**(params["vertically flip"]))
                self.random_hue = r_univariatenormal.layers.RandomHue(**(params["hue"]))
                self.random_jpeg_quality = r_univariatenormal.layers.RandomJpegQuality(**(params["jpeg quality"]))
                self.random_saturation = r_univariatenormal.layers.RandomSaturation(**(params["saturation"]))
            elif domain_randomization.mode == "multivariate normal":
                params = {}
                params["mean_vector"] = domain_randomization.mean_vector
                params["variancecovariance_matrix"] = domain_randomization.variancecovariance_matrix
                params["factors"] = domain_randomization.factors
                params["seed"] = domain_randomization.seed
                self.random_parameters = r_multivariatenormal.layers.RandomParameters(**params)
        else:
            self.domain_randomization = False

        self.field = field


    def call(self, inputs):
        try:
            data = inputs[self.field]
        except KeyError:
            data = inputs
        if self.domain_randomization:
            try:
                dr = self.random_parameters
                data = dr(data)
            except AttributeError:
                data = self.random_brightness(data)
                data = self.random_contrast(data)
                data = self.random_horizontally_flip(data)
                data = self.random_vertically_flip(data)
                data = self.random_hue(data)
                data = self.random_jpeg_quality(data)
                data = self.random_saturation(data)
        base_model_output = self.base_model(data)
        x = self.flatten(base_model_output)
        x = self.dense1(x)
        y = self.dense2(x)
        return y




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