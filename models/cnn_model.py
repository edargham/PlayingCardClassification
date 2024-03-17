import tensorflow as tf
from keras import layers, models, optimizers, losses, metrics
from models.layers.resnet_block import ResidualConvBlock
from models.layers.locnet import LocNet
from typing import List

class CNNModel(models.Model):
  def __init__(
      self,
      image_width: int,
      image_height: int,
      filters_per_layer: List[int],
      kernel_sizes: List[tuple[int, int]],
      num_layers_in_block: int,
      num_classes: int,
      conv_activation: str='relu',
      separable: bool = False,
      dropout_rate: float=0.0,
      **kwargs
  ):
    super(CNNModel, self).__init__(**kwargs)

    if len(kernel_sizes) != len(filters_per_layer):
      raise ValueError(
        """
          Please ensure length the list passed to filters_per_layer is equal to
          the length of the list passed to kernel_sizes.
        """
      )

    self.spatial_transformation = LocNet(
      image_width=image_width,
      image_height=image_height
    )

    self.conv_blocks = []
    for i in range(len(filters_per_layer)):
      self.conv_blocks += [
        ResidualConvBlock(
          filters=filters_per_layer[i],
          kernel_size=kernel_sizes[i],
          num_layers_in_block=num_layers_in_block,
          activation=conv_activation,
          separable=separable
        ),
        layers.MaxPool2D(pool_size=(2, 2))
      ]

    self.lstm_blocks = [
      layers.Dropout(dropout_rate),
      layers.Dense(64, activation="relu", name="dense1"),
      layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25)),
      layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.25))
    ]

    self.dense_block = [
      layers.Dropout(dropout_rate),
      layers.Dense(128, activation='relu'),
      layers.Dropout(dropout_rate),
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dropout(dropout_rate),
      layers.Dense(num_classes, activation='softmax')
    ]

  def call(self, inputs, training=None, mask=None):
    x = inputs
    x = self.spatial_transformation(x)
    
    for l in self.conv_blocks:
      x = l(x)

    x = tf.reshape(x, (-1, tf.shape(x)[3], tf.shape(x)[1]*tf.shape(x)[2]))

    for l in self.lstm_blocks:
      x = l(x)

    for l in self.dense_block:
      x = l(x)

    return x

def build_model(
    optimizer: optimizers.Optimizer,
    loss_fn: losses.Loss,
    model_metrics: List[metrics.Metric],
    image_width: int,
    image_height: int,
    image_channels: int,
    num_classes: int,
    filters: List[int],
    kernel_sizes: List[tuple[int, int]],
    num_layers_in_block: int,
    dropout_rate: float,
    separable: bool=False
):
  model = CNNModel(
    image_width=image_width,
    image_height=image_height,
    filters_per_layer=filters,
    kernel_sizes=kernel_sizes,
    num_layers_in_block=num_layers_in_block,
    dropout_rate=dropout_rate,
    num_classes=num_classes,
    separable=separable
  )
  model.build(input_shape=(None, image_width, image_height, image_channels))
  model.compile(optimizer=optimizer, loss=loss_fn, metrics=model_metrics)
  return model