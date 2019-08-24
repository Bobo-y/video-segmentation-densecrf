import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import numpy as np


def normalize(img):
    return img / 255


def loss(y_true, y_pred):
    logits = y_pred[:, :, :, :1]
    labels = y_true[:, :, :, :1]
    # balance positive and negative samples in an image
    beta = 1 - tf.reduce_mean(labels)
    # first apply sigmoid activation
    predicts = tf.nn.sigmoid(logits)
    # log +epsilon for stable cal
    loss = tf.reduce_mean(
        -1 * (beta * labels * tf.log(predicts + 1e-4) +
              (1 - beta) * (1 - labels) * tf.log(1 - predicts + 1e-4)))
    return loss


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))