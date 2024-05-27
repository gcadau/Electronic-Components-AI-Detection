from tensorflow import keras
import tensorflow as tf
import random
from keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="RandomInverttfr_univariatenormallayers")
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

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomBrightnesstfr_univariatenormallayers")
class RandomBrightness(keras.layers.Layer):
    def __init__(self, mean=0.0, variance=0.15, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): mean=-{0}, variance={sigma_delta} (-> delta sampled from N(0, sigma_delta),
        # with delta value to be applyed to brightness)
        super().__init__(**kwargs)
        self.mean = mean
        self.variance = variance
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                if self.seed is not None:
                    delta = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance, seed=self.seed)
                else:
                    delta = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance)
                ims = []
                for i in range(x.shape[0]):
                    im = x[i,:,:,:]
                    if tf.random.uniform([]) <= self.factor:
                        d = tf.clip_by_value(delta[i], -1, 1)
                        im = tf.image.adjust_brightness(im, d)
                    ims.append(im)
                return tf.stack(ims)
            else: # no batches or eager execution not enabled
                if self.seed is not None:
                    tf.random.set_seed(self.seed)
                    rng = np.random.default_rng(seed=self.seed)
                    delta = rng.normal(self.mean, self.variance, size=1)[0] 
                else:
                    delta = np.random.normal(self.mean, self.variance, size=1)[0]
                if random.random() <= self.factor:
                    d = np.clip(delta, -1, 1)
                    return tf.map_fn(lambda x: tf.image.adjust_brightness(x, d), x)
                else:
                    return x  
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomContrasttfr_univariatenormallayers")
class RandomContrast(keras.layers.Layer):
    def __init__(self, mean=1.25, variance=1, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.variance = variance
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                if self.seed is not None:
                    contrast_factor = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance, seed=self.seed)
                else:
                    contrast_factor = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance)
                ims = []
                for i in range(x.shape[0]):
                    im = x[i,:,:,:]
                    if tf.random.uniform([]) <= self.factor:
                        im = tf.image.adjust_contrast(im, contrast_factor[i])
                    ims.append(im)
                return tf.stack(ims)
            else: # no batches or eager execution not enabled
                if self.seed is not None:
                    tf.random.set_seed(self.seed)
                    rng = np.random.default_rng(seed=self.seed)
                    contrast_factor = rng.normal(self.mean, self.variance, size=1)[0] 
                else:
                    contrast_factor = np.random.normal(self.mean, self.variance, size=1)[0]
                if random.random() <= self.factor:
                    return tf.map_fn(lambda x: tf.image.adjust_contrast(x, contrast_factor), x)
                else:
                    return x 
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

# possibile to define class RandomCrop(keras.layers.Layer), not so useful.


@register_keras_serializable(package="Custom", name="RandomHorizontallyFliptfr_univariatenormallayers")
class RandomHorizontallyFlip(keras.layers.Layer):
    def __init__(self, mean=0.5, variance=0.1, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.variance = variance
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                prob = None
                if self.seed is not None:
                    prob = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance, seed=self.seed)
                else:
                    prob = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance)
                ims = []
                for i in range(x.shape[0]):
                    im = x[i,:,:,:]
                    if tf.random.uniform([]) <= self.factor and prob[i] < 0.5:
                        im = tf.image.flip_left_right(im)
                    ims.append(im)
                return tf.stack(ims)
            else: # no batches or eager execution not enabled
                prob = None
                if self.seed is not None:
                    tf.random.set_seed(self.seed)
                    rng = np.random.default_rng(seed=self.seed)
                    prob = rng.normal(self.mean, self.variance, size=1)[0] 
                else:
                    prob = np.random.normal(self.mean, self.variance, size=1)[0]
                if random.random() <= self.factor and prob < 0.5:
                    return tf.map_fn(lambda x: tf.image.flip_left_right(x), x)
                else:
                    return x 
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomVerticallyFliptfr_univariatenormallayers")
class RandomVerticallyFlip(keras.layers.Layer):
    def __init__(self, mean=0.5, variance=0.1, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.variance = variance
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                prob = None
                if self.seed is not None:
                    prob = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance, seed=self.seed)
                else:
                    prob = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance)
                ims = []
                for i in range(x.shape[0]):
                    im = x[i,:,:,:]
                    if tf.random.uniform([]) <= self.factor and prob[i] < 0.5:
                        im = tf.image.flip_up_down(im)
                    ims.append(im)
                return tf.stack(ims)
            else: # no batches or eager execution not enabled
                prob = None
                if self.seed is not None:
                    tf.random.set_seed(self.seed)
                    rng = np.random.default_rng(seed=self.seed)
                    prob = rng.normal(self.mean, self.variance, size=1)[0] 
                else:
                    prob = np.random.normal(self.mean, self.variance, size=1)[0]
                if random.random() <= self.factor and prob < 0.5:
                    return tf.map_fn(lambda x: tf.image.flip_up_down(x), x)
                else:
                    return x 
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomHuetfr_univariatenormallayers")
class RandomHue(keras.layers.Layer):
    def __init__(self, mean=0.0, variance=0.15, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): mean=-{0}, variance={sigma_delta} (-> delta sampled from N(0, sigma_delta),
        # with delta value to be applyed to hue)
        super().__init__(**kwargs)
        self.mean = mean
        self.variance = variance
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                if self.seed is not None:
                    delta = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance, seed=self.seed)
                else:
                    delta = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance)
                ims = []
                for i in range(x.shape[0]):
                    im = x[i,:,:,:]
                    if tf.random.uniform([]) <= self.factor:
                        d = tf.clip_by_value(delta[i], -1, 1)
                        im = tf.image.adjust_hue(im, d)
                    ims.append(im)
                return tf.stack(ims)
            else: # no batches or eager execution not enabled
                if self.seed is not None:
                    tf.random.set_seed(self.seed)
                    rng = np.random.default_rng(seed=self.seed)
                    delta = rng.normal(self.mean, self.variance, size=1)[0] 
                else:
                    delta = np.random.normal(self.mean, self.variance, size=1)[0]
                if random.random() <= self.factor:
                    return tf.map_fn(lambda x: tf.image.adjust_hue(x, delta), x)
                else:
                    return x 
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomJpegQualitytfr_univariatenormallayers")
class RandomJpegQuality(keras.layers.Layer):
    def __init__(self, mean=60, variance=25, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.sigma = variance
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                if self.seed is not None:
                    jpeg_quality = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance, seed=self.seed)
                else:
                    jpeg_quality = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance)
                ims = []
                for i in range(x.shape[0]):
                    im = x[i,:,:,:]
                    if tf.random.uniform([]) <= self.factor:
                        j = tf.clip_by_value(jpeg_quality[i], 0, 100)
                        im = tf.image.adjust_jpeg_quality(im, j)
                    ims.append(im)
                return tf.stack(ims)
            else: # no batches or eager execution not enabled
                if self.seed is not None:
                    tf.random.set_seed(self.seed)
                    rng = np.random.default_rng(seed=self.seed)
                    jpeg_quality = rng.normal(self.mean, self.variance, size=1)[0] 
                else:
                    jpeg_quality = np.random.normal(self.mean, self.variance, size=1)[0]
                if random.random() <= self.factor:
                    j = np.clip(jpeg_quality, 0, 100)
                    return tf.map_fn(lambda x: tf.image.adjust_jpeg_quality(x, j), x)
                else:
                    return x     
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomSaturationtfr_univariatenormallayers")
class RandomSaturation(keras.layers.Layer):
    def __init__(self, mean=1.25, variance=1.125, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.sigma = variance
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                if self.seed is not None:
                    saturation_factor = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance, seed=self.seed)
                else:
                    saturation_factor = tf.random.normal(shape=[x.shape[0]], mean=self.mean, stddev=self.variance)
                ims = []
                for i in range(x.shape[0]):
                    im = x[i,:,:,:]
                    if tf.random.uniform([]) <= self.factor:
                        s = tf.clip_by_value(saturation_factor[i], 0, float('inf'))
                        im = tf.image.adjust_saturation(im, s)
                    ims.append(im)
                return tf.stack(ims)
            else: # no batches or eager execution not enabled
                if self.seed is not None:
                    tf.random.set_seed(self.seed)
                    rng = np.random.default_rng(seed=self.seed)
                    saturation_factor = rng.normal(self.mean, self.variance, size=1)[0] 
                else:
                    saturation_factor = np.random.normal(self.mean, self.variance, size=1)[0]
                if random.random() <= self.factor:
                    s = np.clip(saturation_factor, 0, float('inf'))
                    return tf.map_fn(lambda x: tf.image.adjust_saturation(x, s), x)
                else:
                    return x     
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"mean": self.mean, "variance": self.variance, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)