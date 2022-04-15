from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Model
from tensorflow import float32


class mmlmodel:
    @staticmethod
    def build(input_shape, classes):
        # Model initialisation
        model = Sequential()

        # First complex of convolutional layers (conv, activation, pooling)
        model.add(Conv2D(filters=20, kernel_size=(5, 5), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second complex of convolutional layers (conv, activation, pooling)
        model.add(Conv2D(filters=50, kernel_size=(5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # First complex of fully connected layers (flatten, dense and activation)
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Classification complex of fully connected layers (dense and activation)
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # We need to return the model structure to make it integrable
        return model
