import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


# =========================
# PATHS
# =========================

train_pre_path = "dataset/train/pre-event"
train_post_path = "dataset/train/post-event"
train_target_path = "dataset/train/target"

val_pre_path = "dataset/val/pre-event"
val_post_path = "dataset/val/post-event"
val_target_path = "dataset/val/target"


# =========================
# LOAD SAMPLE
# =========================

def load_sample(filename, pre_path, post_path, target_path):

    pre = cv2.imread(
        os.path.join(pre_path, filename)
    )

    post = cv2.imread(
        os.path.join(post_path, filename)
    )

    target = cv2.imread(
        os.path.join(target_path, filename),
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
# LOAD DATASET
# =========================

X_train = []
y_train = []

files = os.listdir(train_pre_path)

for filename in files[:500]:

    pre, post, target = load_sample(
        filename,
        train_pre_path,
        train_post_path,
        train_target_path
    )

    combined, processed_target = preprocess_sample(
        pre,
        post,
        target
    )

    X_train.append(combined)
    y_train.append(processed_target)

X_train = np.array(X_train)
y_train = np.array(y_train)

y_train = np.expand_dims(y_train, axis=-1)


# =========================
# VALIDATION DATA
# =========================

X_val = []
y_val = []

val_files = os.listdir(val_pre_path)

for filename in val_files[:100]:

    pre, post, target = load_sample(
        filename,
        val_pre_path,
        val_post_path,
        val_target_path
    )

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
# DICE LOSS
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
# BUILD U-NET
# =========================

def build_unet():

    inputs = layers.Input((256, 256, 6))

    c1 = layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same'
    )(inputs)

    c1 = layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same'
    )(c1)

    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same'
    )(p1)

    c2 = layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same'
    )(c2)

    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(
        128,
        (3, 3),
        activation='relu',
        padding='same'
    )(p2)

    u1 = layers.UpSampling2D((2, 2))(c3)

    u1 = layers.concatenate([u1, c2])

    c4 = layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same'
    )(u1)

    c4 = layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same'
    )(c4)

    u2 = layers.UpSampling2D((2, 2))(c4)

    u2 = layers.concatenate([u2, c1])

    c5 = layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same'
    )(u2)

    c5 = layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same'
    )(c5)

    outputs = layers.Conv2D(
        1,
        (1, 1),
        activation='sigmoid'
    )(c5)

    model = models.Model(inputs, outputs)

    return model


# =========================
# TRAIN MODEL
# =========================

model = build_unet()

model.compile(
    optimizer='adam',
    loss=improved_loss,
    metrics=['accuracy', dice_coefficient]
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=4
)


# =========================
# SAVE MODEL
# =========================

model.save("models/change_detection_model.keras")

print("Training Complete!")