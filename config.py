config = {
  'data_path': '/content',
  'batch_size': 32,
  'epochs': 200,
  'learning_rate': 1.5e-4,
  'dropout_rate': 0.0,
  'filters_per_layer': [32, 64, 64, 128],
  'kernel_size_per_layer': [(3, 3), (3, 3), (5, 5), (5, 5)],
  'separable': False,
  'conv_activation': 'relu',
  'image_width': 224,
  'image_height': 224,
  'image_channels': 3,
  'num_layers_in_block': 1
}
