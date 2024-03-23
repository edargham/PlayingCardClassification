import tensorflow as tf
from keras import callbacks, optimizers, losses
from datetime import datetime
from config import config
from datagen_preprocessing_autoencoders import load_data
from models.autoencoder_model import build_autoencoder  # Import your autoencoder model function

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

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config['learning_rate'],
        decay_steps=239*30,
        decay_rate=0.85,
        staircase=True
    )

    optimizer = optimizers.Adam(learning_rate=lr_schedule)
    loss = losses.MeanSquaredError()  # Use MSE for autoencoder

    model = build_autoencoder(
        optimizer=optimizer,
        loss_fn=loss,
        image_width=config['image_width'],
        image_height=config['image_height'],
        image_channels=config['image_channels'],
        latent_dim=config['latent_dim']
    )

    model.summary()

    train_input = train_data
    train_target = train_data

    val_input = val_data
    val_target = val_data

    model.fit(
        train_input,
        train_target,
        validation_data=(val_input, val_target),
        epochs=config['epochs'],
        callbacks=[
            tensorboard_callback,
            early_stop_callback,
            model_checkpoint
        ]
    )
