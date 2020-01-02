import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import numpy as np
import pydensecrf.densecrf as dcrf


def dense_crf(img, probs, n_labels=2):
    h = probs.shape[0]
    w = probs.shape[1]

    probs = np.expand_dims(probs, 0)
    probs = np.append(1 - probs, probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, n_labels)
    U = -np.log(probs)
    U = U.reshape((n_labels, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    U = U.astype(np.float32)
    d.setUnaryEnergy(U) # Unary

    d.addPairwiseGaussian(sxy=20, compat=3)  #
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q


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