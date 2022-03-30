from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Model
from tensorflow import float32


class LeNet:
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


class MLP1:
    @staticmethod
    def build(input_shape, classes):
        # Input
        input_params = Input(shape=input_shape, name="InputMLP1")

        # First complex of fully connected Dense Layers (dense, activation)
        x = Dense(30, name="Dense1MLP1")(input_params)
        x = Activation("relu", name="Relu1MlP1")(x)

        # Second complex of fully connected Dense Layers (dense, activation)
        x = Dense(15, name="Dense2MLP1")(x)
        x = Activation("relu", name="Relu2MlP1")(x)

        # Third complex of fully connected Dense Layers (dense, activation)
        x = Dense(15, name="Dense3MLP1")(x)
        x = Activation("relu", name="Relu3MlP1")(x)

        # Fourth Complex of Classification Layer (output)
        output = Dense(classes, activation="softmax", name="OutputMLP1")(x)

        return Model(input_params, output)

