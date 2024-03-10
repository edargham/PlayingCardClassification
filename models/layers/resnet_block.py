import tensorflow as tf
from tensorflow.keras import layers
from models.layers.common import conv_layer

class ResidualConvBlock(layers.Layer):
  def __init__(
    self,
    filters: int,
    kernel_size: tuple[int, int],
    activation: str=None,
    separable: bool=False,
    num_layers_in_block: int=3,
    **kwargs
  ):
    super(ResidualConvBlock, self).__init__(**kwargs)
    self.conv_block = []

    for i in range(num_layers_in_block):
      self.conv_block += [
        conv_layer(
          filters=filters,
          kernel_size=kernel_size,
          strides=(1, 1),
          activation=activation,
          padding='same',
          separable=separable
        ),
      ]

    self.residual_conv = conv_layer(
      filters=filters,
      kernel_size=kernel_size,
      strides=(1, 1),
      activation=None,
      padding='same',
      separable=separable
    )

    self.adder = layers.Add()

  def call(self, inputs):
    x = inputs
    for l in self.conv_block:
      x = l(x)
    residual = self.residual_conv(x)
    x = self.adder([x, residual])
