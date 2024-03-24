from keras.layers import Layer
import tensorflow as tf

class KMeansLayer(Layer):
  def __init__(self, num_clusters, **kwargs):
    super(KMeansLayer, self).__init__(**kwargs)
    self.num_clusters = num_clusters

  def build(self, input_shape):
    self.clusters = self.add_weight(
      shape=(self.num_clusters, input_shape[-1]),
      initializer='random_uniform',
      trainable=True,
    )

  def call(self, inputs):
    # Calculate the distance from each input to each cluster center
    distances = tf.norm(tf.expand_dims(inputs, axis=1) - self.clusters, axis=2)
    # Assign each input to the closest cluster
    assignments = tf.argmin(distances, axis=1)
    return tf.gather(self.clusters, assignments)