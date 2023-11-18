from tensorflow import keras



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
    def __init__(self, n_classes, input_shape=(128, 128, 3)):
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

    def call(self, inputs):
        x = self.conv_1(inputs)
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
    def __init__(self, n_classes, input_shape=(128, 128, 3)):
        super(ResNet2__0, self).__init__()
        self.base_model = keras.applications.ResNet152(weights = 'imagenet', include_top = False, input_shape = input_shape)
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(1000, activation='relu')
        self.dense2 = keras.layers.Dense(n_classes, activation='softmax')

        x = self.flatten(self.base_model.output)
        x = self.dense1(x)
        self.y = self.dense2(x)

        self.model = keras.Model(inputs=self.base_model.input, outputs=self.y)


    def call(self, inputs):
        return self.model(inputs)




class ResNet2__1(keras.Model):
    def __init__(self, n_classes, input_shape=(128, 128, 3)):
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


    def call(self, inputs):
        return self.model(inputs)




class ResNet2__0__1(ResNet2__0):
    def __init__(self, n_classes, input_shape=(128, 128, 3)):
        super(ResNet2__0__1, self).__init__(n_classes=n_classes, input_shape=input_shape)
        # To be implemented:
        #       Even after two epochs, validation accuracy arrives near 90%. After 40 epochs the model
        #       comfortably converges. It is possible to reach up to higher accuracies by adding a couple of more fully
        #       connected layers. The successful results with only one hidden fully connected layer mean that ResNet-152
        #       does a pretty good job while extracting features for the classifier even though ImageNet and MNIST
        #       contain fairly distant image samples.

    def call(self, inputs):
        return self.model(inputs)




class ResNet2__1__1(ResNet2__0):
    def __init__(self, n_classes, input_shape=(128, 128, 3)):
        super(ResNet2__1__1, self).__init__(n_classes=n_classes, input_shape=input_shape)
        # To be implemented:
        #       Even after two epochs, validation accuracy arrives near 90%. After 40 epochs the model
        #       comfortably converges. It is possible to reach up to higher accuracies by adding a couple of more fully
        #       connected layers. The successful results with only one hidden fully connected layer mean that ResNet-152
        #       does a pretty good job while extracting features for the classifier even though ImageNet and MNIST
        #       contain fairly distant image samples.

    def call(self, inputs):
        return self.model(inputs)