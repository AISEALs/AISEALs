import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

# model.compile(
#   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#   ....
# )

# Example 1: (batch_size = 1, number of samples = 4)
y_true = [0, 1, 0, 0]
y_pred = [-18.6, 0.51, 2.94, -12.8]
bce = BinaryCrossentropy(from_logits=True)
print(bce(y_true, y_pred).numpy())

y_pred_s1 = tf.nn.softmax(y_pred)
# which one is right?
print(BinaryCrossentropy()(y_true, y_pred_s1).numpy())
y_pred_s2 = tf.sigmoid(y_pred)
print(BinaryCrossentropy()(y_true, y_pred_s2).numpy()) # right

def compute_bce(ys, ys_):
    loss = []
    for y, y_ in zip(ys, ys_):
        if y:
            loss.append(np.log(y_))
        else:
            loss.append(np.log(1-y_))
    return -1/len(ys)*sum(loss)

rs = compute_bce(y_true, y_pred_s2.numpy())

# Example 2: (batch_size = 2, number of samples = 4)
y_true = [[0, 1], [0, 0]]
y_pred = [[-18.6, 0.51], [2.94, -12.8]]
# Using default 'auto'/'sum_over_batch_size' reduction type.
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
bce(y_true, y_pred).numpy()

# Using 'sample_weight' attribute
bce(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()

# Using 'sum' reduction` type.
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
    reduction=tf.keras.losses.Reduction.SUM)
bce(y_true, y_pred).numpy()

# Using 'none' reduction type.
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE)
bce(y_true, y_pred).numpy()

import tensorflow_addons as tfa
y_true = [[1.0], [1.0], [0.0]]
y_pred = [[0.97], [0.91], [0.03]]
tfa.losses.SigmoidFocalCrossEntropy(alpha=1, gamma=0, reduction=tf.keras.losses.Reduction.NONE)(y_true=y_true, y_pred=y_pred).numpy()
# < tf.Tensor: shape = (3,), dtype = float32, numpy = array([6.8532745e-06, 1.9097870e-04, 2.0559824e-05],
#                                                           # dtype=float32) >

BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred).numpy()
