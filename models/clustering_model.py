import tensorflow as tf
from keras import layers, models
from models.layers.kmeans import KMeansLayer
#from metrics.clustering_metrics import Silhouette

class ClusteringModel(models.Model):
  def __init__(
      self,
      cnn_backbone: models.Model, 
      num_clusters: int, 
      **kwargs
    ):
    super(ClusteringModel, self).__init__(**kwargs)
    self.cnn_backbone = cnn_backbone
    self.flatten = layers.Flatten()
    self.num_clusters = num_clusters
    self.kmeans = KMeansLayer(num_clusters)
    # self.silhouette = Silhouette(clusters=num_clusters)

  def call(self, inputs):
    x = inputs
    x = self.cnn_backbone(x)
    feats = self.flatten(x)
    centroids, clusters = self.kmeans(feats)
    return centroids, clusters, feats
  
  @tf.function
  def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape:
      centroids, clusters, feats = self(x)
      loss = self.compiled_loss(centroids, centroids)
    clusters = tf.expand_dims(clusters, axis=-1)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.compiled_metrics.update_state(y, clusters)
    # self.silhouette.update_state(feats, clusters)
    results = {m.name: m.result() for m in self.metrics}
    # results.update({'loss': loss, 'silhouette_score': self.silhouette.result()})
    results.update({'loss': loss })
    return results
  
  @tf.function
  def test_step(self, data):
    x, y = data
    centroids, clusters, feats = self(x)
    clusters = tf.expand_dims(clusters, axis=-1)
    loss = self.compiled_loss(centroids, centroids)
    self.compiled_metrics.update_state(y, clusters)
    # self.silhouette.update_state(feats, clusters)
    results = {m.name: m.result() for m in self.metrics}
    # results.update({'loss': loss, 'silhouette_score': self.silhouette.result()})
    results.update({'loss': loss })
    return results

  # def get_config(self):
  #   return {
  #     # 'cnn_backbone': self.cnn_backbone,
  #     'num_clusters': self.num_clusters
  #   }

  # @classmethod
  # def from_config(cls, config):
  #   return cls(**config)