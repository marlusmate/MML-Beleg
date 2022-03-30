import os
import random
import tensorflow as tf
import json


def get_data_points_list(source_dir, number_points='all', exp_list='all'):
    """
    This function iterative over the given folder and look for all data points (image file and metadata file) and
    returns the list with full file names to those files.
    :param source_dir: directories where data points are to search, sorted by exp_list
    :param exp_list: name of directories with relevant data points
    :param number_points: specifies whether all data points (use tag 'all') must be or only limited number. The later
    helps to test the training pipeline with a small amount of data.
    :return: List of full file names including full paths to files to each data points.
    """

    image_file = []
    metadata_file = []
    source_dir = [os.path.join(source_dir, dir) for dir in os.listdir(source_dir) if
                  any([dir.__contains__(exp) for exp in exp_list])]
    for fdir in source_dir:
        for file in os.listdir(fdir):
            if os.path.isfile(os.path.join(fdir, file)) and file.endswith('.png'):
                filename_image = os.path.join(fdir, file)
                filename = os.path.splitext(file)[0][:-13]
                filename_metadata = os.path.join(fdir, filename + '.json')
                if os.path.isfile(filename_metadata):
                    image_file.append(filename_image)
                    metadata_file.append(filename_metadata)
    if number_points == 'all':
        return list(zip(image_file, metadata_file))
    else:
        return list(zip(image_file, metadata_file))[0:number_points]


def read_image(file):
    """
    This function read an image and returns it as a tensorflow tensor.
    :param file: Full path to the image file
    :return: Image as a tensor or None if the image cannot be read.
    """
    try:
        image_file = tf.io.read_file(file)
        image_data = tf.image.decode_image(image_file)
        return image_data
    except Exception:
        print(f'Image {file} could not be read')
        return None


def read_label(file, no_classes):
    """
    This function reads the label from the metadata file and encodes it according to the one-hot scheme.
    :param file: Full path to the metadata file
    :param no_classes: Total number of all possible classes to allow one-hot encoding
    :return: One-hot encoded label as a tensor
    """
    one_hot_encoder = tf.one_hot(range(no_classes), no_classes)
    with open(file) as f:
        json_content = json.load(f)
        label_int = json_content["flow_regime"]["data"]["value"]
        label = one_hot_encoder[label_int]
        return label


def image_generator(list_data_points, repeats, no_classes, output_image_shape):
    """
    This is a generator that yields a pair of tensors (image and label) every time it is called. Before yielding
    the data points are shuffled and images are preprocessed. Please define preprocessing steps and number of classes
    here.
    :param list_data_points: List of data points (see function get_data_points_list())
    :param repeats: In case of several epochs, number of repeats of dataset can be specified here.
    :return: Yields pair of tensors (preprocessed image and label)
    """
    for repeat in range(repeats):
        for data_point in random.sample(list_data_points, len(list_data_points)):
            image_file = data_point[0]
            image_original = read_image(image_file)
            if image_original is None:
                continue

            # Preprocessing

            # Convert to Gray Scale
            if image_original.shape[2] != 1:
                image_grayscaled = tf.image.rgb_to_grayscale(image_original)
            else:
                image_grayscaled = image_original

            # Crop Image
            offset_width = 0
            target_width = image_grayscaled.shape[0]
            offset_height = image_grayscaled.shape[1] // 5
            target_height = 3 * image_grayscaled.shape[1] // 5

            image_croped = tf.image.crop_to_bounding_box(image_grayscaled, offset_height, offset_width, target_height, target_width)

            # Resize to output_image_shape
            final_image_size = list(output_image_shape)[0:2]
            image_resized = tf.image.resize(image_croped, final_image_size, method='bicubic')

            # Normalize Image
            image_normed = image_resized / 255

            image_data = image_normed

            # Get Label
            label_file = data_point[1]
            label_data = read_label(label_file, no_classes)

            yield image_data, label_data