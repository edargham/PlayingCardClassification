import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from datetime import datetime
from config import config
from datagen_preprocessing import load_data
import numpy as np

from sklearn.metrics import homogeneity_score, completeness_score, silhouette_score
from sklearn.cluster import KMeans

def run_training():
  train_data, val_data, test_data = load_data(
    config['data_path'],
    config['batch_size']
  )

  clusters = train_data.num_classes

  # Load the autoencoder model and consider just the encoder part
  autoencoder = keras.models.load_model('best_autoencoder')
  encoder =  keras.Model(autoencoder.inputs, autoencoder.layers[11].output)
  encoder.trainable = False

  train_feats = np.ravel(encoder.predict(train_data)).reshape(-1, 1)
  val_feats = np.ravel(encoder.predict(val_data)).reshape(-1, 1)
  test_feats = np.ravel(encoder.predict(test_data)).reshape(-1, 1)

  kmeans = KMeans(n_clusters=clusters)
  print('Silhouette Score:', silhouette_score(val_feats, kmeans.fit_predict(val_feats)))


  return kmeans

