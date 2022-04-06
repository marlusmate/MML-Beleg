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


class MMMLP1:
    @staticmethod
    def build_image_extraction(input_shape):
        # Input
        input_image = Input(shape=input_shape, name="InputImageMMMLP1")

        # First complex of convolutional layers (conv, activation, pooling)
        x = Conv2D(filters=20, kernel_size=(7, 7), padding="same", name="conv1MMMLP1")(input_image)
        x = Activation("relu", name="relu1ImgMMMLP1")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1MMMLP1")(x)

        # Second complex of convolutional layers (conv, activation, pooling)
        x = Conv2D(filters=15, kernel_size=(5, 5), padding="same", name="conv2MMMLP1")(x)
        x = Activation("relu", name="relu2ImgMMMLP1")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2MMMLP1")(x)

        # Third complex of convolutional layers (conv, activation, pooling)
        x = Conv2D(filters=10, kernel_size=(5, 5), padding="same", name="conv2MMMLP1")(x)
        x = Activation("relu", name="relu2ImgMMMLP1")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2MMMLP1")(x)

        # Output Layer (flatten)
        output_img = Flatten(name="outputIamgeMMMLP1")(x)

        return Model(inputs=input_image, outputs=output_img, name="ModelImageExtract")

    @staticmethod
    def build_params_extraction(input_shape):
        # Input
        input_params = Input(shape=input_shape, name="InputProcessMMMLP1")

        # First complex of fully connected Dense Layers (dense, activation)
        x = Dense(20, name="dense1ParamMMMLP1")(input_params)
        x = Activation("relu", name="relu1ParamMMMlP1")(x)

        # Second complex of fully connected Dense Layers (dense, activation)
        x = Dense(15, name="dense2ParamMMMLP1")(x)
        x = Activation("relu", name="relu2ParamMMlP1")(x)


        # Output
        output_params = Dense(15, name="OutputParamMMMLP1")(x)

        return Model(inputs=input_params, outputs=output_params, name="ModelParamExtract")

    @staticmethod
    def build_fusion(input_shape_image, input_shape_params, no_classes):
        # Input
        image_in = Input(shape=input_shape_image, name="FrontInputImageMMMLP1")
        params_in = Input(shape=input_shape_params, name="FrontInputParamsMMMLP1")

        # Build Feature extraction blocks (Image, Process Parameters)
        cnn1 = MMMLP1.build_image_extraction(input_shape_image)
        param1 = MMMLP1.build_params_extraction(input_shape_params)

        # First complex of fully connected Layers (concatenate)
        merged = Concatenate()([cnn1(image_in), param1(params_in)])

        # Second complex of fully connected Layers (dense, activation
        x = Dense(15, name="dense1MergedMMMLP1")(merged)
        x = Activation("relu", name="relu1MergedMMMlP1")(x)
        x = Dropout(rate=0.2)(x)


        # Second complex of fully connected Dense Layers (dense, activation)
        x = Dense(15, name="dense2MergedMMMLP1")(x)
        x = Activation("relu", name="relu2MergedMMlP1")(x)
        x = Dropout(rate=0.2)(x)

        # Output
        output = Dense(no_classes, activation="sigmoid", name="OutputMergedMMMLP1")(x)

        return Model(inputs=(image_in, params_in), outputs=output, name="ModelMultimodal")
