from tensorflow import keras
import tensorflow as tf
import random
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="RandomInverttfr_uniformlayers")
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

@register_keras_serializable(package="Custom", name="RandomBrightnesstfr_uniformlayers")
class RandomBrightness(keras.layers.Layer):
    def __init__(self, lower=-0.2, upper=0.2, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): lower=-{max_delta}, upper={max_delta} (-> delta sampled from [-delta, delta],
        # with delta value to be applyed to brightness)
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                if self.seed is not None:
                    delta = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper, seed=self.seed)
                else:
                    delta = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper)
                ims = []
                for i in range(x.shape[0]):
                    im = x[i,:,:,:]
                    if tf.random.uniform([]) <= self.factor:
                        im = tf.image.adjust_brightness(im, delta[i])
                    ims.append(im)
                return tf.stack(ims)
            else: # no batches or eager execution not enabled
                if self.seed is not None:
                    tf.random.set_seed(self.seed)
                    rng = np.random.default_rng(seed=self.seed)
                    delta = rng.uniform(self.lower, self.upper, size=1)[0] 
                else:
                    delta = np.random.uniform(self.lower, self.upper, size=1)[0] 
                if random.random() <= self.factor:
                    return tf.map_fn(lambda x: tf.image.adjust_brightness(x, delta), x)
                else:
                    return x  
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"lower": self.lower, "upper": self.upper, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomContrasttfr_uniformlayers")
class RandomContrast(keras.layers.Layer):
    def __init__(self, lower=0, upper=2.5, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                if self.seed is not None:
                    contrast_factor = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper, seed=self.seed)
                else:
                    contrast_factor = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper)
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
                    contrast_factor = rng.uniform(self.lower, self.upper, size=1)[0] 
                else:
                    contrast_factor = np.random.uniform(self.lower, self.upper, size=1)[0] 
                if random.random() <= self.factor:
                    return tf.map_fn(lambda x: tf.image.adjust_contrast(x, contrast_factor), x)
                else:
                    return x  
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"lower": self.lower, "upper": self.upper, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

# possibile to define class RandomCrop(keras.layers.Layer), not so useful.


@register_keras_serializable(package="Custom", name="RandomHorizontallyFliptfr_uniformlayers")
class RandomHorizontallyFlip(keras.layers.Layer):
    def __init__(self, lower=0, upper=1, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): lower=0, upper=1 -> same probability to get
        # a value greater/lower than 0.5 (-> threshold between flip/don't flip)
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                prob = None
                if self.seed is not None:
                    prob = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper, seed=self.seed)
                else:
                    prob = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper)
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
                    prob = rng.uniform(self.lower, self.upper, size=1)[0] 
                else:
                    prob = np.random.uniform(self.lower, self.upper, size=1)[0] 
                if random.random() <= self.factor and prob < 0.5:
                    return tf.map_fn(lambda x: tf.image.flip_left_right(x), x)
                else:
                    return x  
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"lower": self.lower, "upper": self.upper, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomVerticallyFliptfr_uniformlayers")
class RandomVerticallyFlip(keras.layers.Layer):
    def __init__(self, lower=0, upper=1, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): lower=0, upper=1 -> same probability to get
        # a value greater/lower than 0.5 (-> threshold between flip/don't flip)
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                prob = None
                if self.seed is not None:
                    prob = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper, seed=self.seed)
                else:
                    prob = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper)
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
                    prob = rng.uniform(self.lower, self.upper, size=1)[0] 
                else:
                    prob = np.random.uniform(self.lower, self.upper, size=1)[0] 
                if random.random() <= self.factor and prob < 0.5:
                    return tf.map_fn(lambda x: tf.image.flip_up_down(x), x)
                else:
                    return x  
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"lower": self.lower, "upper": self.upper, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomHuetfr_uniformlayers")
class RandomHue(keras.layers.Layer):
    def __init__(self, lower=-0.2, upper=0.2, seed=None, factor=0.9, **kwargs):
        # usual behaviour (not mandatory): lower=-{max_delta}, upper={max_delta} (-> delta sampled from [-delta, delta],
        # with delta value to be applyed to hue)
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                if self.seed is not None:
                    delta = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper, seed=self.seed)
                else:
                    delta = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper)
                ims = []
                for i in range(x.shape[0]):
                    im = x[i,:,:,:]
                    if tf.random.uniform([]) <= self.factor:
                        im = tf.image.adjust_hue(im, delta[i])
                    ims.append(im)
                return tf.stack(ims)
            else: # no batches or eager execution not enabled
                if self.seed is not None:
                    tf.random.set_seed(self.seed)
                    rng = np.random.default_rng(seed=self.seed)
                    delta = rng.uniform(self.lower, self.upper, size=1)[0] 
                else:
                    delta = np.random.uniform(self.lower, self.upper, size=1)[0] 
                if random.random() <= self.factor:
                    return tf.map_fn(lambda x: tf.image.adjust_hue(x, delta), x)
                else:
                    return x  
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"lower": self.lower, "upper": self.upper, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomJpegQualitytfr_uniformlayers")
class RandomJpegQuality(keras.layers.Layer):
    def __init__(self, lower=20, upper=100, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                if self.seed is not None:
                    jpeg_quality = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper, seed=self.seed)
                else:
                    jpeg_quality = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper)
                ims = []
                for i in range(x.shape[0]):
                    im = x[i,:,:,:]
                    if tf.random.uniform([]) <= self.factor:
                        im = tf.image.adjust_jpeg_quality(im, jpeg_quality[i])
                    ims.append(im)
                return tf.stack(ims)
            else: # no batches or eager execution not enabled
                if self.seed is not None:
                    tf.random.set_seed(self.seed)
                    rng = np.random.default_rng(seed=self.seed)
                    jpeg_quality = rng.uniform(self.lower, self.upper, size=1)[0] 
                else:
                    jpeg_quality = np.random.uniform(self.lower, self.upper, size=1)[0] 
                if random.random() <= self.factor:
                    return tf.map_fn(lambda x: tf.image.adjust_jpeg_quality(x, jpeg_quality), x)
                else:
                    return x  
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"lower": self.lower, "upper": self.upper, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)

@register_keras_serializable(package="Custom", name="RandomSaturationtfr_uniformlayers")
class RandomSaturation(keras.layers.Layer):
    def __init__(self, lower=0, upper=2, seed=None, factor=0.9, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.factor = factor


    def call(self, x, training=None):
        if training:
            if x.shape[0] is not None: 
                if self.seed is not None:
                    saturation_factor = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper, seed=self.seed)
                else:
                    saturation_factor = tf.random.uniform(shape=[x.shape[0]], minval=self.lower, maxval=self.upper)
                ims = []
                for i in range(x.shape[0]):
                    im = x[i,:,:,:]
                    if tf.random.uniform([]) <= self.factor:
                        im = tf.image.adjust_saturation(im, saturation_factor[i])
                    ims.append(im)
                return tf.stack(ims)
            else: # no batches or eager execution not enabled
                if self.seed is not None:
                    tf.random.set_seed(self.seed)
                    rng = np.random.default_rng(seed=self.seed)
                    saturation_factor = rng.uniform(self.lower, self.upper, size=1)[0] 
                else:
                    saturation_factor = np.random.uniform(self.lower, self.upper, size=1)[0] 
                if random.random() <= self.factor:
                    return tf.map_fn(lambda x: tf.image.adjust_saturation(x, saturation_factor), x)
                else:
                    return x  
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"lower": self.lower, "upper": self.upper, "seed": self.seed, "factor": self.factor})
        return config

	@classmethod
	def from_config(cls, config):
		return cls(**config)