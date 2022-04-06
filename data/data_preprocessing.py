import os
import shutil
import numpy as np
from tensorflow.keras.utils import array_to_img
from PIL import Image
from data_import.data_import import get_data_points_list, read_image, read_json, preprocess_image
from data_import.data_clean import change_data_dir

# Path to raw data
data_raw_folder = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes_me/03_raw_data'

# Path to preprocessed data
data_prepro_folder = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes_me/02_data'

# Get List of data points
data_points = get_data_points_list(data_raw_folder)

# Crop Matrix
crop_matrix = {"02-08": [394, 1617],"02-09": [153, 1777],"02-22": [257, 1727],"02-25": [257, 1727],"03-09": [255, 1798]}

# Preprocess
image_output_shape = (128, 128, 1)
for data_point in data_points[:10]:
    img_file = data_point[0]
    proc_file = data_point[1]

    img = read_image(img_file)
    file = read_json(data_point)

    if any(file is None for file in [img, file]) is True:
        continue

    exp_date = [date for date in crop_matrix.keys() if date in img_file]

    img = preprocess_image(img,crop_box=crop_matrix[exp_date[0]], output_image_shape=image_output_shape)

    print("Typ vorverarbeitetes Bild-array: ", type(img))

    img = array_to_img(np.array(img))
    print("Typ abzuspeicherendes Bild-array: ", type(img))

    # new path
    data_point_name = img_file.split('/')[-1]
    img_file = os.path.join(data_prepro_folder,data_point_name)
    proc_file_new = os.path.join(data_prepro_folder, data_point_name.replace("_camera_frame", ""))

    # Move Files
    img.save(img_file)
    shutil.copy(proc_file, proc_file_new.replace("png", "json"))
    print("Moved file ", data_point_name, " to ", data_prepro_folder)












# Save in new dir