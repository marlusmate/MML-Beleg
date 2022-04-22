import tensorflow as tf
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from data_import.data_import import data_generator, read_picklelist
from data_import.data_clean import convert_to_binary, get_pred_proba
from models.model import mmlmodel
from sklearn.metrics import roc_curve


# Dataset Parameters
param_list = ["stirrer_rotational_speed", "gas_flow_rate", "temperature", "fill_level"]
output_proc_shape = (len(param_list),)
batch_size = 15
output_img_shape = (128, 128, 1)
no_classes = 3
no_epochs = 1

# Model Name
model_type = "LeNet20x50"

# Paths
model_path = '../training/results'
model_name = f'/trained_model{model_type}'
model_checkpoint_path = f'../training/results/checkpoints{model_type}' + '/checkpoint-0020.ckpt'
image_list = '../training/image-points-test.pickle'
param_list = '../training/param-points-test.pickle'

# Data Generator
data_points = read_picklelist(image_path=image_list, param_path=param_list)
shuffled_data_points = random.sample(data_points, len(data_points))

# Get Distribution
lb_test = []
for data_point in data_points_test:
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

# Formulate Predictions
pred = model.predict(dataset_test_batched, batch_size=batch_size)
pred_tf = tf.constant(np.argmax(pred, axis=-1))

# Load true Labels
lb = []
for data_point in data_points:
    with open(data_point[1]) as f:
        json_content = json.load(f)
        label_int = json_content["flow_regime"]["data"]["value"]
        lb.append(label_int)
lb_tf = tf.constant(lb)
one_hot_encoder = tf.one_hot(range(no_classes), no_classes)
lb_onehot = [one_hot_encoder(label) for label in lb]
# Confusion Matrix
# Rows: True / Label
# Columns: Predicition
cf_mat = tf.math.confusion_matrix(lb_tf, pred_tf, num_classes=no_classes)
row_sums = tf.math.reduce_sum(input_tensor= cf_mat, axis=1)
cf_mat_norm = cf_mat / row_sums
plt.matshow(cf_mat, cmap=plt.cm.gray)
conf1 = plt.gcf()
plt.show()
conf1.savefig("../Figures/ConfusionMatrix")

# Get Metrics - Precision, Recall, F1
# Gotta love multiclass problems
# Get binary Labels (axis=0 == columns, axis=1 == rows
y_pred_n = tf.reduce_sum(cf_mat, axis=0)
y_true_n = tf.reduce_sum(cf_mat, axis=1)

# get one vs rest labels
ovr_0_pred, ovr_0_true = convert_to_binary(lb_pred=pred_tf, lb_true=lb_tf, label=0)
ovr_1_pred, ovr_1_true = convert_to_binary(lb_pred=pred_tf, lb_true=lb_tf, label=1)
ovr_2_pred, ovr_2_true = convert_to_binary(lb_pred=pred_tf, lb_true=lb_tf, label=2)

# Calc Precision, Recall, F1
# Label 0
precision_0 = Precision()
precision_0.update_state(y_true=ovr_0_true, y_pred=ovr_0_pred)

recall_0 = Recall()
recall_0.update_state(y_true=ovr_0_true, y_pred=ovr_0_pred)

# Label 1
precision_1 = Precision()
precision_1.update_state(y_true=ovr_1_true, y_pred=ovr_1_pred)

recall_1 = Recall()
recall_1.update_state(y_true=ovr_1_true, y_pred=ovr_1_pred)

# Label 2
precision_2 = Precision()
precision_2.update_state(y_true=ovr_2_true, y_pred=ovr_2_pred)

recall_2 = Recall()
recall_2.update_state(y_true=ovr_2_true, y_pred=ovr_2_pred)

# Plot Metrics
metrics_names = ["Precision", "Recall"]
class_0 = [precision_0.result().numpy(), recall_0.result().numpy()]
class_1 = [precision_1.result().numpy(), recall_1.result().numpy()]
class_2 = [precision_2.result().numpy(), recall_2.result().numpy()]

x_axis = np.arange(len(metrics_names))

plt.bar(x_axis -0.3, class_0, width=0.2, label = '0')
plt.bar(x_axis -0.1, class_1, width=0.2, label = '1')
plt.bar(x_axis +0.1, class_2, width=0.2, label = '2')
plt.xticks(x_axis, metrics_names)
plt.legend()
bar1 = plt.gcf()
plt.show()
bar1.savefig("../Figures/Barplot_Precsion-Recall")


# ROC, AUC, log loss?
# y_pred value for true class of instance needed
y_true_proba = get_pred_proba(pred_proba=pred, y_true=lb_tf)

fpr_0, tpr_0, thresholds_0 = roc_curve(ovr_0_true, y_true_proba)
fpr_1, tpr_1, thresholds_1 = roc_curve(ovr_1_true, y_true_proba)
fpr_2, tpr_2, thresholds_2 = roc_curve(ovr_2_true, y_true_proba)

plt.plot(fpr_0, tpr_0, label=0)
plt.plot(fpr_1, tpr_1, label=1)
plt.plot(fpr_2, tpr_2, label=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.legend()
roc1 = plt.gcf()
plt.show()
roc1.savefig("../Figures/Roc_curves")

# Calc AUC
auc_0 = tf.keras.metrics.AUC()
auc_0.update_state(ovr_0_true, ovr_0_pred)
auc_0_val = auc_0.result().numpy()


# Save Metrics to json-file





