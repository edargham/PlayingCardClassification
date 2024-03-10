from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

def load_data(
  data_base_dir: str,
  batch_size: int
) -> tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
  train_datagen = ImageDataGenerator(
    rescale=1./255,
  )

  training_set = train_datagen.flow_from_directory(
    directory=f'{data_base_dir}/train',
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=True,
  )

  val_datagen = ImageDataGenerator(
    rescale=1./255,
  )

  validation_set = val_datagen.flow_from_directory(
    directory=f'{data_base_dir}/valid',
    target_size=(224, 224),
    batch_size=batch_size,
  )

  test_datagen = ImageDataGenerator(
    rescale=1./255,
  )

  test_set = test_datagen.flow_from_directory(
    directory=f'{data_base_dir}/test',
    target_size=(224, 224),
    batch_size=batch_size,
  )

  return training_set, validation_set, test_set