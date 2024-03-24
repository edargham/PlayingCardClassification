import tensorflow as tf
from keras import losses

class KMeansLoss(losses.Loss):
  def __init__(self, clusters: int, **kwargs):
    super().__init__(**kwargs)
    self.clusters = clusters

  def call(self, y_true, y_pred):
    # y_true is ignored in this loss function
    return tf.reduce_mean(tf.square(tf.norm(y_pred - self.clusters, axis=-1)))