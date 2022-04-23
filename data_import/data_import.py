import os
import random
import tensorflow as tf
import json
import numpy as np
import pickle


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
    for file in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, file)) and file.endswith('.png'):
            filename_image = os.path.join(source_dir, file)
            filename = os.path.splitext(file)[0][:-13]
            filename_metadata = os.path.join(source_dir, filename + '.json')
            if os.path.isfile(filename_metadata):
                image_file.append(filename_image)
                metadata_file.append(filename_metadata)
    if number_points == 'all':
        return list(zip(image_file, metadata_file))
    else:
        return list(zip(image_file, metadata_file))[0:number_points]


def load_json(data_point):
    """
    This loads a json file as a dictionary every time it is called.
    :param data_point: data point (see function get_data_points_list())
    :return: json file as dict
    """
    json_file = data_point[1]
    with open(json_file) as f:
        json_content = json.load(f)

    return json_content


def read_picklelist(image_path, param_path):
    """
    This functions loads lists of data points, saved as pickle and zips them together for a list of data points
    :param image_path: path to list (pickle) of image data-points
    :param param_path: path to list (pickle) of parameters data-points
    :return: list of tuples of data points
    """
    with open(image_path, 'rb') as f:
        data_points_image = pickle.load(f)
    with open(param_path, 'rb') as f:
        data_points_param = pickle.load(f)

    return list(zip(data_points_image, data_points_param))


def read_json(data_point, param_list='all'):
    """
    This returns the process data of a json file (from a dict) as a list (See function load_json()).
    :param data_point: data point (see function get_data_points())
    :param param_list: list of parameters to be extracted
    :return: list of parameter values, in the same order (no labels)
    """
    params = []
    json_content = load_json(data_point)
    if param_list == 'all':
        param_list = ["stirrer_rotational_speed", "gas_flow_rate", "temperature", "fill_level"]
    if not all(["value" in json_content[param]["data"] for param in param_list]):
        return None

    params = [json_content[param]["data"]["value"] for param in param_list]

    return params


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


def preprocess_image(image_file, crop_box, output_image_shape):
    # Preprocessing
    # Convert to Gray Scale
    if image_file.shape[2] != 1:
        image_grayscaled = tf.image.rgb_to_grayscale(image_file)
    else:
        image_grayscaled = image_file
    print("Shape image_grayscaled: ", image_grayscaled.shape, "; Type: ", type(image_grayscaled))

    # Crop Box, Size
    crop_points = [crop_box[0], 0, crop_box[1], image_grayscaled.shape[0]]
    box = tf.constant([crop_points])
    box_ind = tf.constant(0)
    crop_size = tf.constant(output_image_shape[:-1])

    # Crop and
    image_cropped = tf.image.crop_to_bounding_box(image_grayscaled, crop_points[0], crop_points[1], crop_points[2]-crop_points[0],
                                                  crop_points[3])
    # Resize
    final_image_size = list(output_image_shape)[0:2]
    image_resized = tf.image.resize(image_cropped, final_image_size, method='bicubic')

    # Normalize Image
    image_normed = image_resized / 255

    return image_normed


def data_generator(list_data_points, repeats, no_classes, output_image_shape, param_list):
    for repeat in range(repeats):
        for data_point in random.sample(list_data_points, len(list_data_points)):
            image_file = data_point[0]
            label_file = data_point[1]

            image_preprocessed = read_image(image_file)
            proc_list = read_json(data_point, param_list)

            if any(file is None for file in [image_preprocessed, proc_list]) is True:
                continue

            #image_data = preprocess_image(image_original, output_image_shape)
            proc_data = tf.convert_to_tensor(proc_list)

            label_data = read_label(label_file, no_classes)

            yield (image_preprocessed, proc_data), label_data