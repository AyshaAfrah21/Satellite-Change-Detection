import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    confusion_matrix
)


# =========================
# PATHS
# =========================

val_pre_path = "dataset/val/pre-event"
val_post_path = "dataset/val/post-event"
val_target_path = "dataset/val/target"


# =========================
# LOAD SAMPLE
# =========================

def load_sample(filename):

    pre = cv2.imread(
        os.path.join(val_pre_path, filename)
    )

    post = cv2.imread(
        os.path.join(val_post_path, filename)
    )

    target = cv2.imread(
        os.path.join(val_target_path, filename),
        cv2.IMREAD_GRAYSCALE
    )

    return pre, post, target


# =========================
# PREPROCESSING
# =========================

def preprocess_sample(pre, post, target):

    pre = cv2.resize(pre, (256, 256))
    post = cv2.resize(post, (256, 256))
    target = cv2.resize(target, (256, 256))

    pre = pre / 255.0
    post = post / 255.0

    target = target.astype(np.float32)

    combined = np.concatenate([pre, post], axis=-1)

    return combined, target


# =========================
# DICE FUNCTIONS
# =========================

def dice_coefficient(y_true, y_pred):

    smooth = 1e-6

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return (
        2. * intersection + smooth
    ) / (
        tf.reduce_sum(y_true_f)
        +
        tf.reduce_sum(y_pred_f)
        +
        smooth
    )


def dice_loss(y_true, y_pred):

    return 1 - dice_coefficient(y_true, y_pred)


# =========================
# WEIGHTED BCE
# =========================

def weighted_bce(y_true, y_pred):

    epsilon = 1e-7

    y_pred = tf.clip_by_value(
        y_pred,
        epsilon,
        1. - epsilon
    )

    weight_for_1 = 15.0
    weight_for_0 = 1.0

    bce = -(
        weight_for_1 * y_true * tf.math.log(y_pred)
        +
        weight_for_0 * (1 - y_true) * tf.math.log(1 - y_pred)
    )

    return tf.reduce_mean(bce)


def improved_loss(y_true, y_pred):

    wbce = weighted_bce(y_true, y_pred)

    dloss = dice_loss(y_true, y_pred)

    return wbce + dloss


# =========================
# LOAD MODEL
# =========================

model = load_model(
    "models/change_detection_model.keras",
    custom_objects={
        "improved_loss": improved_loss,
        "dice_coefficient": dice_coefficient
    }
)


# =========================
# LOAD VALIDATION DATA
# =========================

X_val = []
y_val = []

files = os.listdir(val_pre_path)

for filename in files[:100]:

    pre, post, target = load_sample(filename)

    combined, processed_target = preprocess_sample(
        pre,
        post,
        target
    )

    X_val.append(combined)
    y_val.append(processed_target)

X_val = np.array(X_val)
y_val = np.array(y_val)

y_val = np.expand_dims(y_val, axis=-1)


# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X_val)

pred_masks = (predictions > 0.3).astype(np.uint8)


# =========================
# FLATTEN
# =========================

y_true = (y_val > 0.5).astype(np.uint8).flatten()

y_pred = pred_masks.flatten()


# =========================
# METRICS
# =========================

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

iou = jaccard_score(
    y_true,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(y_true, y_pred)

print("\n===== RESULTS =====")

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("IoU:", iou)

print("\nConfusion Matrix:\n")
print(cm)