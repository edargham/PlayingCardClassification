from config import config
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import numpy as np
def load_data(
    data_base_dir: str,
    batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.3, 1.0),
        horizontal_flip=True,
        vertical_flip=True,
    )

    training_set = train_datagen.flow_from_directory(
        directory=f'{data_base_dir}/train',
        target_size=(config['image_height'], config['image_width']),
        batch_size=batch_size,
        shuffle=True,
        class_mode=None  # No labels needed for autoencoder
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
    )

    validation_set = val_datagen.flow_from_directory(
        directory=f'{data_base_dir}/valid',
        target_size=(config['image_height'], config['image_width']),
        batch_size=batch_size,
        class_mode=None  # No labels needed for autoencoder
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255,
    )

    test_set = test_datagen.flow_from_directory(
        directory=f'{data_base_dir}/test',
        target_size=(config['image_height'], config['image_width']),
        batch_size=batch_size,
        class_mode=None  # No labels needed for autoencoder
    )

    # train_input = np.concatenate([training_set[i][0] for i in range(len(training_set))])
    # val_input = np.concatenate([validation_set[i][0] for i in range(len(validation_set))])
    # test_input = np.concatenate([test_set[i][0] for i in range(len(test_set))])
    train_input = np.concatenate(training_set)
    val_input = np.concatenate(validation_set)
    test_input = np.concatenate(test_set)

    return train_input, val_input, test_input
