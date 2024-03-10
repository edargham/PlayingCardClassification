from keras_tuner import RandomSearch, HyperParameters
from tensorflow.keras import callbacks, optimizers, losses, metrics
from datetime import datetime
from config import config
from datagen_preprocessing import load_data
from models.cnn_model import build_model, CNNModel

def run_tuning():
  def build_hypermodel(hp: HyperParameters)->CNNModel:
    num_layers_per_block = hp.Choice('num_layers_per_block', [1, 3, 5])
    separable = hp.Boolean('separable')
    learning_rate = hp.Float('learning_rate', min_value=1e-3, max_value=1e-2, step=2e-3)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)

    lr_schedule = optimizers.schedules.ExponentialDecay(
      initial_learning_rate=learning_rate,
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

    hypermodel = build_model(
      optimizer=optimizer,
      loss_fn=loss,
      model_metrics=tracking_metrics,
      image_width=config['image_width'],
      image_height=config['image_height'],
      image_channels=config['image_channels'],
      num_classes=train_data.num_classes,
      filters=config['filters_per_layer'],
      kernel_sizes=config['kernel_size_per_layer'],
      num_layers_in_block=num_layers_per_block,
      dropout_rate=dropout_rate,
      separable=separable
    )

    return hypermodel


  logdir = 'logs/tuning/' + datetime.now().strftime("%Y%m%d-%H%M%S")

  tensorboard_callback = callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
  early_stop_callback = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    mode='min',
    restore_best_weights=True
  )
  model_checkpoint = callbacks.ModelCheckpoint(
    'best_model_tuned',
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

  tuner = RandomSearch(
    build_hypermodel,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2,
    directory='tuning_results',
    project_name='model_tuning'
  )

  tuner.search(
    train_data,
    validation_data=val_data,
    epochs=config['epochs'],
    callbacks=[
      tensorboard_callback,
      early_stop_callback,
      model_checkpoint
    ]
  )

  best_hps = tuner.get_best_hyperparameters()
  print(
    f"""
      The hyperparameter search is complete.\n
      Optimal hyperparameters:\n
      {best_hps}
    """
  )
  print('Training on estimated optimal hyperparameters:')

  model = tuner.hypermodel.build(best_hps)

  history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=config['epochs'],
    callbacks=[
      tensorboard_callback,
      early_stop_callback,
      model_checkpoint
    ]
  )

  return model, history




