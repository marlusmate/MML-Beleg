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
        x = Conv2D(filters=10, kernel_size=(5, 5), padding="same", name="conv3MMMLP1")(x)
        x = Activation("relu", name="relu3ImgMMMLP1")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool3MMMLP1")(x)

        # First complex of dense layer (flatten, dense)
        x = Flatten(name="flattenIamgeMMMLP1")(x)
        x = Dense(20, name="dense1ImgMMMLP1")(x)

        # Output Layer (activation)
        output_img = Activation("relu", name="outputImageMMMlP1")(x)

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
    def build_fusion(input_shape_image, input_shape_params, classes):
        # Input
        image_in = Input(shape=input_shape_image, name="FrontInputImageMMMLP1")
        params_in = Input(shape=input_shape_params, name="FrontInputParamsMMMLP1")

        # Build Feature extraction blocks (Image, Process Parameters)
        cnn1 = mmlmodel.build_image_extraction(input_shape_image)
        param1 = mmlmodel.build_params_extraction(input_shape_params)

        # EARLY FUSION
        # First complex of fully connected Layers (concatenate)
        earlyfusion1_merged = Concatenate()([cnn1(image_in), param1(params_in)])

        # First complex of fully connected Layers (dense, activation, dropout)
        x = Dense(15, name="dense1MergedMMMLP1")(earlyfusion1_merged)
        x = Activation("relu", name="relu1MergedMMMlP1")(x)
        x = Dropout(rate=0.4)(x)

        # Second complex of fully connected Dense Layers (dense, activation, dropout)
        x = Dense(15, name="dense2MergedMMMLP1")(x)
        x = Activation("relu", name="relu2MergedMMlP1")(x)
        x = Dropout(rate=0.4)(x)

        # Auxiliary Output for Early Fusion Features
        earlyfusion1_output = Dense(classes, activation="sigmoid", name="OutputMergedMMMLP1")(x)

        # LATE FUSION
        latefusion1_merged = Concatenate()([cnn1(image_in), x, param1(params_in)])

        # Third complex of fully connected Layers (dense, activation, dropout)
        x = Dense(15, name="dense1MergedMMMLP2")(latefusion1_merged)
        x = Activation("relu", name="relu1MergedMMMlP2")(x)
        x = Dropout(rate=0.4)(x)

        # Second complex of fully connected Dense Layers (dense, activation, dropout)
        x = Dense(15, name="dense2MergedMMMLP2")(x)
        x = Activation("relu", name="relu2MergedMMlP2")(x)
        x = Dropout(rate=0.4)(x)

        # Output
        final_output = Dense(classes, activation="sigmoid", name="OutputMergedMMMLP2")(x)

        return Model(inputs=(image_in, params_in), outputs=(final_output, earlyfusion1_output), name="ModelMultimodal")

    @staticmethod
    def build_model(input_shape_image, input_shape_params, classes):
        # Input
        image_in = Input(shape=input_shape_image, name="FrontInputImageMMMLP1")
        params_in = Input(shape=input_shape_params, name="FrontInputParamsMMMLP1")

        # Build Feature extraction blocks (Image, Process Parameters)
        cnn1 = mmlmodel.build_image_extraction(input_shape_image)
        param1 = mmlmodel.build_params_extraction(input_shape_params)

        # Build Early Fusion block (Image -, Process Features)
        earlyfusion1 = mmlmodel.build_fusion(input_shape_image, input_shape_params, no_classes)

        # Build late Fusion block (Image -, Process -, Higher Level Features)
        merged = Concatenate()([cnn1(image_in), earlyfusion1((image_in, params_in)),param1(params_in)])

        # Second complex of fully connected Layers (dense, activation
        x = Dense(15, name="dense1MergedMMMLP2")(merged)
        x = Activation("relu", name="relu1MergedMMMlP2")(x)
        x = Dropout(rate=0.4)(x)

        # Second complex of fully connected Dense Layers (dense, activation)
        x = Dense(15, name="dense2MergedMMMLP2")(x)
        x = Activation("relu", name="relu2MergedMMlP2")(x)
        x = Dropout(rate=0.4)(x)

        # Output
        output = Dense(no_classes, activation="sigmoid", name="OutputMergedMMMLP2")(x)

        return Model(inputs=(image_in, params_in), outputs=output, name="ModelMultimodal-hybridfusion")


