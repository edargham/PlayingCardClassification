from config import config
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator


def create_datagen(
  directory: str, 
  batch_size: int, 
  shuffle: bool=False, 
  augment: bool=False
) -> tuple[DirectoryIterator, int]:
  datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.3, 1.0),
    horizontal_flip=True,
    vertical_flip=True
  ) if augment else ImageDataGenerator(rescale=1./255)

  generator = datagen.flow_from_directory(
    directory,
    target_size=(config['image_height'], config['image_width']),
    batch_size=batch_size,
    shuffle=shuffle,
    class_mode=None  # No labels needed for autoencoder
  )

  def data_generator():
    while True:
      data = next(generator)
      yield data, data

  return data_generator(), generator.samples

def load_data(
  data_base_dir: str,
  batch_size: int
) -> tuple[DirectoryIterator, int, DirectoryIterator, int, DirectoryIterator, int]:
  training_set, train_samples = create_datagen(
    f'{data_base_dir}/train', 
    batch_size,
    shuffle=True, 
    augment=True
  )
  validation_set, val_samples = create_datagen(f'{data_base_dir}/valid', batch_size)
  test_set, test_samples = create_datagen(f'{data_base_dir}/test', batch_size)
  return (
    training_set, 
    train_samples,
    validation_set, 
    val_samples, 
    test_set, 
    test_samples
  )
