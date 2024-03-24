from sklearn.metrics import homogeneity_score, completeness_score
import tensorflow as tf
from keras import metrics


class Homogeneity(metrics.Metric):
  def __init__(
      self, 
      clusters: int,
      name='homogeneity', 
      **kwargs
    ):
    super(Homogeneity, self).__init__(name=name, **kwargs)
    self.homogeneity = self.add_weight(name='hom', initializer='zeros')
    self.clusters = clusters

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.argmax(y_true, axis=-1)
    # Calculate the distance from each prediction to each cluster center
    distances = tf.norm(tf.expand_dims(y_pred, axis=1) - self.clusters, axis=2)
    # Assign each prediction to the closest cluster
    y_pred = tf.argmin(distances, axis=1)
    homogeneity = tf.py_function(homogeneity_score, (y_true, y_pred), tf.float32)
    self.homogeneity.assign(homogeneity)

  def result(self):
    return self.homogeneity

class Completeness(metrics.Metric):
  def __init__(
      self, 
      clusters: int,
      name='completeness', 
      **kwargs
    ):
    super(Completeness, self).__init__(name=name, **kwargs)
    self.completeness = self.add_weight(name='comp', initializer='zeros')
    self.clusters = clusters

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.argmax(y_true, axis=-1)
    # Calculate the distance from each prediction to each cluster center
    distances = tf.norm(tf.expand_dims(y_pred, axis=1) - self.clusters, axis=2)
    # Assign each prediction to the closest cluster
    y_pred = tf.argmin(distances, axis=1)
    completeness = tf.py_function(completeness_score, (y_true, y_pred), tf.float32)
    self.completeness.assign(completeness)


  def result(self):
    return self.completeness