import tensorflow as tf
import json
import random
import pickle
import numpy as np
from tensorflow.keras.optimizers import Adam
from data_import.data_import import data_generator
from models.model import mmlmodel

# Dataset Parameters
param_list = ["stirrer_rotational_speed", "gas_flow_rate", "temperature", "fill_level"]
output_proc_shape = (len(param_list),)
batch_size = 15
output_img_shape = (128, 128, 1)
no_classes = 3
no_epochs = 1

# Model Name
model_name = "LeNet20x50"

# Paths
model_path = '../training/results'
model_trained = f'/results/{model_name}/trained_model'
model_checkpoint_path = f'../training/results/{model_name}' + '/checkpoints/checkpoint-0010.ckpt'
data_list = '../data/data-points-test.pickle'

# Data Generator
with open(data_list, 'rb') as file:
    # Call load method to deserialze
    data_points = pickle.load(file)
shuffled_data_points = random.sample(data_points, len(data_points))

# Get Distribution
lb_test = []
for data_point in data_points:
    with open(data_point[1]) as f:
        json_content = json.load(f)
        lb_test.append(json_content["flow_regime"]["data"]["value"])
print("Test Instanzen:\n-----------------------")
print("Klasse 0: ", lb_test.count(0), "\nKlasse 1: ", lb_test.count(1), "\nKlasse 2: ", lb_test.count(2))

output_signature = (tf.TensorSpec(shape=output_img_shape, dtype=tf.float32),
                    tf.TensorSpec(shape=(no_classes), dtype=tf.bool))

data_gen_test = data_generator(data_points, repeats=no_epochs, no_classes=3, output_image_shape=output_img_shape, param_list=param_list)
dataset_test = tf.data.Dataset.from_generator(lambda: data_gen_test, output_signature=output_signature)
dataset_test_batched = dataset_test.batch(batch_size)

# Build Model
opt = Adam()
model = mmlmodel.build(input_shape=output_img_shape, classes=no_classes)
model.load_weights(model_checkpoint_path)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.summary()

# Save Predictions
pred = model.predict(dataset_test_batched, batch_size=batch_size)
np.save(f"y_pred_{model_name}.npy", pred)
#with open(f"y_pred_{model_name}.pickle", 'wb') as f:
    #pickle.dump(pred, f)
pred_tf = tf.constant(np.argmax(pred, axis=-1))
print("Vorhersagen(", len(pred), f") abgespeichert - y_pred_{model_name}.pickle")

