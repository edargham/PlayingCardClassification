import tensorflow as tf
from keras import layers

def conv_layer(
    filters:int,
    kernel_size: tuple[int, int]=(3, 3),
    strides: tuple[int, int]=(1, 1),
    activation=None,
    padding='same',
    separable: bool=False
) -> layers.SeparableConv2D | layers.Conv2D:
  if separable:
    return layers.SeparableConv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      # activation=activation,
      padding=padding
    )
  else:
    return layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      # activation=activation,
      padding=padding
    )