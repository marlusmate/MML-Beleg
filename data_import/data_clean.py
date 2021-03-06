import shutil
import json
import os
import numpy as np

from data_import.data_import import read_json, load_json


def change_flow_regime(data_point, label_mat):
    """
    This Changes the label of the data-point, according to label_mat.
    Overwrites old json-file
    :param data_point: single data point (see function get_data_points_list()).
    :param label_mat: contains matrix of set-points and associated labels [gasflow, rpm, label].
    :return: None(, overwritten label in json-file)
    """
    file = load_json(data_point)
    prc_data = read_json(data_point)
    for i in np.arange(len(label_mat)):
        if prc_data is None:
            continue

        if label_mat[i][1] == prc_data[0] and int(prc_data[1]) in np.arange(label_mat[i][0] - 2, label_mat[i][0] + 2):
            file["flow_regime"]["data"]["value"] = label_mat[i][2]
            out_file = open(data_point[1], 'w+')
            json.dump(file, out_file, indent=4)
        else:
            continue
    return


def change_data_dir(data_point, target_dir):
    """
    Moves data_point (image, metadata(json)) to target_dir
    :param data_point:
    :param target_dir:
    :return:
    """
    shutil.move(data_point[0], target_dir)
    shutil.move(data_point[1], target_dir)
    print(data_point, "moved to ", target_dir)
    return


def sort_data_points(data_points, set_points, num):
    """
    This returns two lists of data points, the original list is divided by
    checking whether param value is located within setpoints.
    Loads each json file (default params, see function read_json() )
    :param data_points: original list of data points
    :param set_points: Dict with np.arrange of setpoints (dict.values())
    :param num: param from json file for decision criterium
    :return: Two np.arrays of the data_points
    """
    data_points_setpoint = []
    data_points_trans = []
    for data_point in data_points:
        prc = read_json(data_point)
        if prc is None:
            continue
        if not any(int(prc[num]) in set_points[i] for i in set_points):
            data_points_trans.append(data_point)
            continue

        data_points_setpoint.append(data_point)

    data_points_setpoint = np.array(data_points_setpoint)
    data_points_trans = np.array(data_points_trans)

    return data_points_setpoint, data_points_trans


def convert_to_binary(lb_pred, lb_true, label):
    """
    Converts multiclass labels (0,1,2) to binary, one-vs-rest labels
    :param lb_pred: list of predicted multiclass labels
    :param lb_true: list of true multiclass label
    :param label: label to be the "one" against the other labels
    :return: list of binary label
    """
    lb_pred_new = []
    lb_true_new = []
    for lb in lb_pred:
        if lb == label:
            lb_pred_new.append(1)
        else:
            lb_pred_new.append(0)

    for lb in lb_true:
        if lb == label:
            lb_true_new.append(1)
        else:
            lb_true_new.append(0)

    return lb_pred_new, lb_true_new


def get_pred_proba(pred_proba, y_true):
    """
    Selects predicted probalities of true classes from prediction set
    :param pred_proba: One hot Label Probabilities
    :param y_true: Sparse True Label
    :return: list of predicted probabilites for true classes
    """
    pred = []
    for i in np.arange(len(pred_proba)):
        proba = pred_proba[i][y_true[i]]
        pred.append(proba)

    return pred



