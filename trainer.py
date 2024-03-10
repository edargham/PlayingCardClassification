from tensorflow.keras import callbacks, optimizers, losses, metrics
from datetime import datetime
from config import config
from datagen_preprocessing import load_data
from models.cnn_model import build_model

def run_training():
  logdir = 'logs/training/' + datetime.now().strftime("%Y%m%d-%H%M%S")

  tensorboard_callback = callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
  early_stop_callback = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    mode='min',
    restore_best_weights=True
  )
  model_checkpoint = callbacks.ModelCheckpoint(
    'best_model',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    save_format='tf',
    verbose=1
  )

  train_data, val_data, test_data = load_data(
    config['data_path'],
    config['batch_size']
  )

  lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=config['learning_rate'],
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
  )

  optimizer = optimizers.Adam(learning_rate=lr_schedule)
  loss = losses.CategoricalCrossentropy()

  tracking_metrics = [
    metrics.CategoricalAccuracy(name='acc'),
    metrics.Precision(name='precision'),
    metrics.Recall(name='recall')
  ]

  model = build_model(
    optimizer=optimizer,
    loss_fn=loss,
    model_metrics=tracking_metrics,
    image_width=config['image_width'],
    image_height=config['image_height'],
    image_channels=config['image_channels'],
    num_classes=train_data.num_classes,
    filters=config['filters_per_layer'],
    kernel_sizes=config['kernel_size_per_layer'],
    num_layers_in_block=config['num_layers_in_block'],
    dropout_rate=config['dropout_rate'],
    separable=config['separable']
  )

  model.summary()

  model.fit(
    train_data,
    validation_data=val_data,
    epochs=config['epochs'],
    callbacks=[
      tensorboard_callback,
      early_stop_callback,
      model_checkpoint
    ]
  )


