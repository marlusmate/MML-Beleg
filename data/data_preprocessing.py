import os
import shutil
from PIL import Image
from data_import.data_import import get_data_points_list, read_image, preprocess_image
from data_import.data_clean import change_data_dir

# Path to raw data
data_raw_folder = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes_me/03_raw_data'

# Path to preprocessed data
data_prepro_folder = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes_me/02_data'

# Get List of data points
data_points = get_data_points_list(data_raw_folder)


# Preprocess
for data_point in data_points:
    img_file = data_point[0]
    proc_file = data_point[1]

    img = read_image(img_file)
    img = preprocess_image(img)
    print(type("Typ vorverarbeitetes Bild-array: ", img))

    img = Image.fromarray(img)
    print(type("Typ abzuspeicherendes Bild-array: ", img))

    # new path
    data_point_name = img_file.split('/')[-1]
    img_file = os.path.join(data_prepro_folder,data_point_name)
    proc_file_new = os.path.join(data_prepro_folder, data_point_name)

    # Move Files
    img.save(os.path.join(img_file))
    shutil.copy(proc_file, proc_file_new)
    print("Moved file ", data_point_name, " to ", data_prepro_folder)












# Save in new dir