# Import
import os
import random
import shutil
from data_import.data_import import get_data_points_list
# Directories
source_dir =
target_dir =

# Get data_points, shuffeld list
data_points = get_data_points_list(source_dir)
shuffled_data_points = random.sample(data_points, len(data_points))
dataset_len = len(shuffled_data_points)

# Split Final Test set
split_ratio = [0.7, 0.1]

# Training-, Validation Set
no_train_points = int(split_ratio[0] / sum(split_ratio) * dataset_len)
data_points_train = shuffled_data_points[:no_train_points]

# Test Set
no_test_points = int(split_ratio[0] / sum(split_ratio) * dataset_len)
data_points_test = shuffled_data_points[no_train_points: no_train_points + no_test_points]

# Print
print("Anzahl Daten-Punkte:\n---------------------------\n ")
print("Training, Validation: ", no_train_points," von ", dataset_len, "(",no_train_points/dataset_len,"%)")
print("Finales Testen: ", no_test_points," von ", dataset_len, "(",no_test_points/dataset_len,"%)")

# Move Data points
for data_point in data_points_test:
    # adjust path
    img_name = data_point[0].split('/')[-1]
    file_name = img_name.replace("_camera_frame", "")
    file_name = file_name.replace("png", "json")

    img_path = os.path.join(target_dir,img_name)
    file_path = os.path.join(target_dir, file_name)

    # Move Files
    shutil.move(data_point[0], img_path) # Image
    shutil.move(data_point[1], file_path) # Json-File
    print("Moved file ", data_point[0], " to ", img_path)