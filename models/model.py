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
        # Input
        input_params = Input(shape=input_shape, name="InputMLP1")

        # First complex of fully connected Dense Layers (dense, activation)
        x = Dense(30, name="Dense1MLP1")(input_params)
        x = Activation("relu", name="Relu1MlP1")(x)

        # Second complex of fully connected Dense Layers (dense, activation)
        x = Dense(15, name="Dense2MLP1")(x)
        x = Activation("relu", name="Relu2MlP1")(x)

        # Third complex of fully connected Dense Layers (dense, activation)
        x = Dense(7, name="Dense3MLP1")(x)
        x = Activation("relu", name="Relu3MlP1")(x)

        # Fourth complex of fully connected Dense Layers (dense, activation)
        x = Dense(15, name="Dense4MLP1")(x)
        x = Activation("relu", name="Relu4MlP1")(x)

        # Fourth Complex of Classification Layer (output)
        output = Dense(classes, activation="softmax", name="OutputMLP1")(x)

        return Model(input_params, output)
