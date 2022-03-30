import os
from models.model import MLP1
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import json

from data_import.data_import import get_data_points_list, process_generator

# Disable CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Path to data
data_folder = 'F:\Data'

# Path to where save model
checkpoint_path = './results/checkpointsMLP/checkpoint-{epoch:04d}.ckpt'

# Path to save logs for tensorboard
tensorboard_log_folder = './resultsMLP/tensorboard'

# Dataset parameters
exp_list = ["exp_2022-02-25_exp03"]
param_list = ["stirrer_rotational_speed", "gas_flow_rate", "temperature", "fill_level"]
no_classes = 3
split_ratio = [0.9, 0.1, 0.0]
output_proc_shape = (len(param_list),)
output_img_shape = (128, 128, 1)

# Training hyper parameters
no_epochs = 5
batch_size = 1
init_lr = 0.001

# Get list of data points
data_points = get_data_points_list(data_folder, number_points='all', exp_list=exp_list)
shuffled_data_points = random.sample(data_points, len(data_points))
dataset_len = len(shuffled_data_points)

# Dataset output siganture
output_signature = (tf.TensorSpec(shape=output_proc_shape, dtype=tf.float32),
                        tf.TensorSpec(shape=(no_classes), dtype=tf.bool))

# Training dataset
no_train_points = int(split_ratio[0] / sum(split_ratio) * dataset_len)
data_points_train = shuffled_data_points[:no_train_points]

data_gen_train = process_generator(data_points_train, no_epochs, no_classes, param_list)
dataset_train = tf.data.Dataset.from_generator(lambda: data_gen_train, output_signature=output_signature)
dataset_train_batched = dataset_train.batch(batch_size)

# Validation dataset
no_val_points = int(split_ratio[1] / sum(split_ratio) * dataset_len)
data_points_val = shuffled_data_points[no_train_points:no_train_points + no_val_points]

data_gen_val = process_generator(data_points_train, no_epochs, no_classes, param_list)
dataset_val = tf.data.Dataset.from_generator(lambda: data_gen_val, output_signature=output_signature)
dataset_val_batched = dataset_val.batch(batch_size)

# Test dataset
no_test_points = int(split_ratio[2] / sum(split_ratio) * dataset_len)
data_points_test = shuffled_data_points[no_train_points + no_val_points:]

data_gen_test = process_generator(data_points_test, no_epochs, no_classes, param_list)
dataset_test = tf.data.Dataset.from_generator(lambda: data_gen_test, output_signature=output_signature)
dataset_test_batched = dataset_test.batch(batch_size)


# Model compilation
opt = Adam(learning_rate=init_lr, decay=init_lr / no_epochs)
model = MLP1.build(input_shape=output_proc_shape, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

# Callback to save model after each batch
save_every_epoch_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=no_train_points // batch_size)
model.save_weights(checkpoint_path.format(epoch=0))

# Callback to stop training after no performance decrease
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                           patience=10,
                                                           restore_best_weights=True)
# Callback to write logs onto tensorboard
# To run tensorboard execute command: tensorboard --logdir training/results/tensorboard
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_folder,
                                             histogram_freq=1,
                                             write_graph=True,
                                             write_images=True,
                                             update_freq='epoch',
                                             profile_batch=2,
                                             embeddings_freq=1)

# Train model
history = model.fit(dataset_train_batched,
                    epochs=no_epochs,
                    batch_size=batch_size,
                    steps_per_epoch=no_train_points // batch_size,
                    validation_data=dataset_val_batched,
                    validation_steps=no_val_points // batch_size,
                    callbacks=[save_every_epoch_callback, early_stopping_callback, tb_callback],
                    )

# Saving model
model.save('./results/trained_modelMLP1')

# Print and save on disk model training history
with open('./results/reportMLP1.json', 'w', encoding='utf-8') as f:
    json.dump(history.history, f, ensure_ascii=False, indent=4)
