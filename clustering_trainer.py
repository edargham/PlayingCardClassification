import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from datetime import datetime
from config import config
from datagen_preprocessing import load_data

from models.clustering_model import ClusteringModel
from losses.kmeans_loss import KMeansLoss
from metrics.clustering_metrics import Completeness, Homogeneity

def run_training():
  logdir = 'logs/training/' + datetime.now().strftime("%Y%m%d-%H%M%S")

  tensorboard_callback = callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
  early_stop_callback = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    mode='min',
    restore_best_weights=True
  )
  model_checkpoint = callbacks.ModelCheckpoint(
    'best_cluster_model',
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

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1.0,#config['learning_rate'],
    decay_steps=(train_data.samples//config['batch_size'])*30,
    decay_rate=0.85,
    staircase=True
  )

  # Load the autoencoder model and consider just the encoder part
  autoencoder = keras.models.load_model('best_autoencoder')
  encoder =  keras.Model(autoencoder.inputs, autoencoder.layers[11].output)
  encoder.trainable = False

  model = ClusteringModel(
    cnn_backbone=encoder,
    num_clusters=train_data.num_classes
  )

  model.build((None, config['image_height'], config['image_width'], config['image_channels']))

  model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=KMeansLoss(clusters=train_data.num_classes),
    metrics=[
      Completeness(clusters=train_data.num_classes),
      Homogeneity(clusters=train_data.num_classes)
    ]
  )

  model.summary()

  model.fit(
    train_data,
    validation_data=val_data,
    epochs=config['epochs'],
    callbacks=[tensorboard_callback, early_stop_callback]
  )

  return model

