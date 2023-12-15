from tensorflow import keras
import tensorflow as tf
import numpy as np


class RandomInvert(keras.layers.Layer):

    def __init__(self, factor=0.5, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.seed = seed


    @staticmethod
    def __random_invert_img(x, p=0.5):
        if tf.random.uniform([]) < p:
            x_tf = (255 - x)
        else:
            x_tf = x
        return x_tf

    def call(self, x, training=None):
        if training:
            return self.__random_invert_img(x, self.factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'factor': self.factor})
        return config


class Brightness(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        if par<-1:
            par = -1
        if par>1:
            par = 1
        self.delta = par


    def call(self, x, training=None):
        if training:
            return tf.image.adjust_brightness(x, self.delta)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'delta': self.delta})
        return config


class Contrast(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        self.contrast_factor = par


    def call(self, x, training=None):
        if training:
            return tf.image.adjust_contrast(x, self.contrast_factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'contrast_factor': self.contrast_factor})
        return config


# possibile to define class Crop(keras.layers.Layer), not so useful.


class HorizontallyFlip(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        self.prob = par


    def call(self, x, training=None):
        if training:
            if self.prob < 0.5:
                return tf.image.flip_left_right(x)
            else:
                return x
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'prob': self.prob})
        return config


class VerticallyFlip(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        self.prob = par


    def call(self, x, training=None):
        if training:
            if self.prob < 0.5:
                return tf.image.flip_up_down(x)
            else:
                return x
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'prob': self.prob})
        return config


class Hue(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        if par<-1:
            par = -1
        if par>1:
            par = 1
        self.delta = par


    def call(self, x, training=None):
        if training:
            return tf.image.adjust_hue(x, self.delta)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'delta': self.delta})
        return config


class JpegQuality(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        if par<0:
            par = 0
        if par>100:
            par = 100
        self.jpeg_quality = par


    def call(self, x, training=None):
        if training:
            ims = []
            for i in range(x.shape[0]):
                im = x[i,:,:,:]
                im = tf.image.adjust_jpeg_quality(im, self.jpeg_quality)
                ims.append(im)
            x = tf.stack(ims)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'jpeg_quality': self.jpeg_quality})
        return config


class Saturation(keras.layers.Layer):
    def __init__(self, par, **kwargs):
        super().__init__(**kwargs)
        if par<0:
            par = 0
        self.saturation_factor = par


    def call(self, x, training=None):
        if training:
            return tf.image.adjust_saturation(x, self.saturation_factor)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'saturation_factor': self.saturation_factor})
        return config





class RandomParameters(keras.layers.Layer):
    def __init__(self, mean_vector=None, variancecovariance_matrix=None, seed=None, factors=None, **kwargs):
        super().__init__(**kwargs)

        if factors is None:
            factors = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        if mean_vector is None:
            mean_vector = [0, 1.25, 0.5, 0.5, 0, 60, 1.25]
        if variancecovariance_matrix is None:
            variancecovariance_matrix = [
                [0.15, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0.1, 0, 0, 0, 0],
                [0, 0, 0, 0.1, 0, 0, 0],
                [0, 0, 0, 0, 0.15, 0, 0],
                [0, 0, 0, 0, 0, 25, 0],
                [0, 0, 0, 0, 0, 0, 1.125]
            ]

        self.factor = factors
        self.mean = np.array(mean_vector)
        self.variance = np.array(variancecovariance_matrix)
        self.seed = seed


    def call(self, x, training=None):
        if training:
            self.randoms = []
            self.do = []
            if self.seed is not None:
                tf.random.set_seed(self.seed)
                rng = np.random.default_rng(seed=self.seed)
                random_parameters = rng.multivariate_normal(self.mean, self.variance, size=x.shape[0])
            else:
                random_parameters = np.random.multivariate_normal(self.mean, self.variance, size=x.shape[0])

            random_brightness = []
            for i in range(x.shape[0]):
                if tf.random.uniform([]) > self.factor[0]:
                    params = {'par': random_parameters[i][0]}
                    random_brightness.append(Brightness(**params))
                else:
                    random_brightness.append(NoneTransformation())
            self.randoms.append(random_brightness)
            random_contrast = []
            for i in range(x.shape[0]):
                if tf.random.uniform([]) > self.factor[1]:
                    params = {'par': random_parameters[i][1]}
                    random_contrast.append(Contrast(**params))
                else:
                    random_contrast.append(NoneTransformation())
            self.randoms.append(random_contrast)
            random_horizontallyFlip = []
            for i in range(x.shape[0]):
                if tf.random.uniform([]) > self.factor[2]:
                    params = {'par': random_parameters[i][2]}
                    random_horizontallyFlip.append(HorizontallyFlip(**params))
                else:
                    random_horizontallyFlip.append(NoneTransformation())
            self.randoms.append(random_horizontallyFlip)
            random_verticallyFlip = []
            for i in range(x.shape[0]):
                if tf.random.uniform([]) > self.factor[3]:
                    params = {'par': random_parameters[i][3]}
                    random_verticallyFlip.append(VerticallyFlip(**params))
                else:
                    random_verticallyFlip.append(NoneTransformation())
            self.randoms.append(random_verticallyFlip)
            random_hue = []
            for i in range(x.shape[0]):
                if tf.random.uniform([]) > self.factor[4]:
                    params = {'par': random_parameters[i][4]}
                    random_hue.append(Hue(**params))
                else:
                    random_hue.append(NoneTransformation())
            self.randoms.append(random_hue)
            random_jpegQuality = []
            for i in range(x.shape[0]):
                if tf.random.uniform([]) > self.factor[5]:
                    params = {'par': random_parameters[i][5]}
                    random_jpegQuality.append(JpegQuality(**params))
                else:
                    random_jpegQuality.append(NoneTransformation())
            self.randoms.append(random_jpegQuality)
            random_saturation = []
            for i in range(x.shape[0]):
                if tf.random.uniform([]) > self.factor[6]:
                    params = {'par': random_parameters[i][6]}
                    random_saturation.append(Saturation(**params))
                else:
                    random_saturation.append(NoneTransformation())
            self.randoms.append(random_saturation)

            for i in range(len(self.factor)):
                ran = self.randoms[i]
                ims = []
                for j in range(x.shape[0]):
                    im = x[j,:,:,:]
                    im = ran[j](im, training=True)
                    ims.append(im)
                x = tf.stack(ims)

        return x

    def get_config(self):
        config = super().get_config()
        keys = ['brightness', 'contrast', 'horizontallyFlip', 'verticallyFlip', 'hue', 'jpegQuality', 'saturation']
        values = [[r.get_config if r is not None else '-' for r in self.randoms[i]] for i in range(7)]
        config_dict = {'random_{}_params'.format(key): value for key, value in zip(keys, values)}
        config.update(config_dict)
        return config



class NoneTransformation():
    def __int__(self):
        pass

    def call(self, x):
        return x
