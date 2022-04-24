import tensorflow as tf
import json
import random
import pickle
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
model_type = "HybridFusion1"

# Paths
data_list = '../data/data-points-test.pickle'

# Data Generator
with open(data_list, 'rb') as file:
    # Call load method to deserialze
    data_points = pickle.load(file)
shuffled_data_points = random.sample(data_points, len(data_points))
print("Datenpunkte '", data_list, "' geladen, n=", len(data_points))

# Load Predictions
pred = np.load(f"y_pred_{model_type}.npy")
pred_finaloutput = pred[0]
pred_tf = tf.convert_to_tensor(np.argmax(pred_finaloutput, axis=-1))

# Load true Labels
lb = []
for data_point in shuffled_data_points:
    with open(data_point[1]) as f:
        json_content = json.load(f)
        label_int = json_content["flow_regime"]["data"]["value"]
        lb.append(label_int)
lb_tf = tf.constant(lb)
with open("y_true.json", 'wb') as f:
    pickle.dump(lb_tf, f)
one_hot_encoder = tf.one_hot(range(no_classes), no_classes)
lb_onehot = [one_hot_encoder[label] for label in lb]


# Confusion Matrix
# Rows: True / Label
# Columns: Predicition
cf_mat = tf.math.confusion_matrix(lb_tf, pred_tf, num_classes=no_classes)
row_sums = tf.math.reduce_sum(input_tensor= cf_mat, axis=1)
cf_mat_norm = cf_mat / row_sums
#plt.matshow(cf_mat, cmap=plt.cm.gray)
plt.imshow(cf_mat_norm, interpolation='nearest', cmap=plt.cm.gray)
classNames = ['0','1', '2']
plt.title('Flow Regime Classification Normed Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
plt.show()
conf1 = plt.gcf()
plt.show()
conf1.savefig(f"../Figures/{model_type}/ConfusionMatrix")
print("CF:\n", cf_mat)
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
bar1.savefig(f"../Figures/{model_type}/Barplot_Precsion-Recall")


# ROC, AUC, log loss?
# y_pred value for true class of instance needed
y_true_proba = get_pred_proba(pred_proba=pred_finaloutput, y_true=lb_tf)

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
roc1.savefig(f"../Figures/{model_type}/Roc_curves")

# Calc AUC
auc_0 = tf.keras.metrics.AUC()
auc_0.update_state(ovr_0_true, ovr_0_pred)
auc_0_val = auc_0.result().numpy()


# Save Metrics to json-file
metrics = {'precision_0': precision_0, 'precision_1': precision_1, 'precision_2': precision_2,
           'recall_0': recall_0, 'recall_1': recall_1, 'recall_2': recall_2}




