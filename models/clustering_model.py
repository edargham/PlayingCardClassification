import tensorflow as tf
from keras import layers, models
from models.layers.kmeans import KMeansLayer

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

  def call(self, inputs):
    x = inputs
    x = self.cnn_backbone(x)
    x = self.flatten(x)
    x = self.kmeans(x)
    return x

  # def get_config(self):
  #   return {
  #     # 'cnn_backbone': self.cnn_backbone,
  #     'num_clusters': self.num_clusters
  #   }

  # @classmethod
  # def from_config(cls, config):
  #   return cls(**config)