from sklearn.metrics import homogeneity_score, completeness_score, silhouette_score
import tensorflow as tf
from keras import metrics


class Homogeneity(metrics.Metric):
  def __init__(self, name='homogeneity', **kwargs):
    super(Homogeneity, self).__init__(name=name, **kwargs)
    self.homogeneity = self.add_weight(name='hom', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    def update_homogeneity(y_true, y_pred):
      return homogeneity_score(y_true.numpy(), y_pred.numpy())
    y_true = tf.reduce_max(y_true, axis=-1)
    y_pred = tf.reduce_max(y_pred, axis=-1)
    homogeneity = tf.py_function(update_homogeneity, (y_true, y_pred), tf.float32)
    self.homogeneity.assign(homogeneity)

  def result(self):
    return self.homogeneity

class Completeness(metrics.Metric):
  def __init__(self, name='completeness', **kwargs):
    super(Completeness, self).__init__(name=name, **kwargs)
    self.completeness = self.add_weight(name='comp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    def update_completeness(y_true, y_pred):
      return completeness_score(y_true.numpy(), y_pred.numpy())
    y_true = tf.reduce_max(y_true, axis=-1)
    y_pred = tf.reduce_max(y_pred, axis=-1)
    completeness = tf.py_function(update_completeness, (y_true, y_pred), tf.float32)
    self.completeness.assign(completeness)

  def result(self):
    return self.completeness
  
# class Silhouette(metrics.Metric):
#   def __init__(self, clusters: int, name='silhouette_score', **kwargs):
#     super(Silhouette, self).__init__(name=name, **kwargs)
#     self.silhouette_score = self.add_weight(name='ss', initializer='zeros')
#     self.clusters = clusters

#   def update_state(self, X, y_pred, sample_weight=None):
#     def update_silhouette(X, y_pred):
#       print(y_pred.numpy())
#       return silhouette_score(X.numpy(), y_pred.numpy())
#     y_pred = tf.cast(y_pred, tf.float32)
#     y_pred = tf.reduce_max(y_pred, axis=-1)
#     silhouette = tf.py_function(update_silhouette, (X, y_pred), tf.float32)
#     self.silhouette_score.assign(silhouette)

#   def result(self):
#     return self.silhouette_score