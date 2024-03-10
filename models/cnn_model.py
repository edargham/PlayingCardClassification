from tensorflow.keras import layers, models, optimizers, losses, metrics
from models.layers.resnet_block import ResidualConvBlock

from typing import List

class CNNModel(models.Model):
  def __init__(
      self,
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
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2))
      ]

    self.flatten = layers.Flatten()

    self.dense_block = [
      layers.Dropout(dropout_rate),
      layers.Dense(512, activation='relu'),
      layers.Dropout(dropout_rate),
      layers.Dense(256, activation='relu'),
      layers.Dense(256, activation='relu'),
      layers.Dropout(dropout_rate),
      layers.Dense(num_classes, activation='softmax')
    ]

  def call(self, inputs, training=None, mask=None):
    x = inputs
    for l in self.conv_blocks:
      x = l(x)
    x = self.flatten(x)
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