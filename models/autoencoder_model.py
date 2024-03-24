from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


def build_autoencoder(optimizer, loss_fn, image_width, image_height, image_channels, latent_dim):
  # Encoder
  input_img = Input(shape=(image_height, image_width, image_channels))
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  encoded = MaxPooling2D((2, 2), padding='same')(x)

  # Decoder
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  decoded = Conv2D(image_channels, (3, 3), activation='sigmoid', padding='same')(x)

  autoencoder = Model(input_img, decoded)
  autoencoder.compile(optimizer=optimizer, loss=loss_fn)

  return autoencoder
