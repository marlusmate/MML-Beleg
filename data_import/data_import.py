import os
import random
import tensorflow as tf
import json
import pandas as pd
import numpy as np


def get_exp_list():
    """
    (not pretty) Extracts Experiments-folder-names that are to be used for training.
    Which Exps are used is defined within csv / Excel file

    :return: list of Names of exp-folders that have data for training/test purpose
    """
    df = pd.read_csv("../data/ExpTab.csv", delimiter=";")
    exp_list = []
    print("Ausgewählte Experimente:")
    for exp in np.arange(len(df)):
        if df.iloc[exp]["in-use"] == 1:
            exp_string = "exp_2022-" + df.iloc[exp]["Datum"] + "_" + df.iloc[exp]["ExpNr"]
            if not exp_list.__contains__(exp_string):
                exp_list.append(exp_string)
                print(exp_string)
    print("Anzahl Setpoints: ", len(exp_list))

    return exp_list


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

    if exp_list == 'all':
        exp_list = os.listdir(source_dir)
    image_file = []
    metadata_file = []
    source_dirs = [os.path.join(source_dir, dir) for dir in os.listdir(source_dir) if
                   exp_list.__contains__(dir)]
    for fdir in source_dirs:
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


def preprocess_image(image_file, output_image_shape):
    # Preprocessing
    # Convert to Gray Scale
    if image_file.shape[2] != 1:
        image_grayscaled = tf.image.rgb_to_grayscale(image_file)
    else:
        image_grayscaled = image_file

    # Crop Image
    offset_width = 0
    target_width = image_grayscaled.shape[0]
    offset_height = image_grayscaled.shape[1] // 5
    target_height = 4 * image_grayscaled.shape[1] // 5

    image_croped = tf.image.crop_to_bounding_box(image_grayscaled, offset_height, offset_width, target_height,
                                                 target_width)

    # Resize to output_image_shape
    final_image_size = list(output_image_shape)[0:2]
    image_resized = tf.image.resize(image_croped, final_image_size, method='bicubic')

    # Normalize Image
    image_normed = image_resized / 255

    return image_normed


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

            image_data = preprocess_image(image_original, output_image_shape)

            # Get Label
            label_file = data_point[1]
            label_data = read_label(label_file, no_classes)

            yield image_data, label_data


def process_generator(list_data_points, repeats, no_classes, param_list):
    """
    This is a generator that yields a pair of tensors (process data and label) every time it is called. Before yielding
    the data points are shuffled. (Please define preprocessing steps and number of classes here.)
    :param list_data_points: List of data points (see function get_data_points_list())
    :param repeats: In case of several epochs, number of repeats of dataset can be specified here.
    :return: Yields pair of tensors (process data parameters  and label)
    """
    for repeat in range(repeats):
        for data_point in random.sample(list_data_points, len(list_data_points)):
            proc_list = read_json(data_point, param_list)
            if proc_list is None:
                continue
            else:
                proc_data = tf.convert_to_tensor(proc_list)

                label_file = data_point[1]
                label_data = read_label(label_file, no_classes)

                yield proc_data, label_data


def data_generator(list_data_points, repeats, no_classes, output_image_shape, param_list):
    for repeat in range(repeats):
        for data_point in random.sample(list_data_points, len(list_data_points)):
            image_file = data_point[0]
            label_file = data_point[1]

            image_original = read_image(image_file)
            proc_list = read_json(data_point, param_list)

            if any(file is None for file in [image_original, proc_list]) is True:
                continue

            image_data = preprocess_image(image_original, output_image_shape)
            proc_data = tf.convert_to_tensor(proc_list)

            label_data = read_label(label_file, no_classes)

            yield (image_data, proc_data), label_data
