from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras_preprocessing.image import ImageDataGenerator
from network import Seg
from utils import *
import os
import json
import warnings
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=tf_config)
ktf.set_session(session)


config = json.load("config.json")
batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
total_training_imgs = config['train']['totoal_training_imgs']
total_val_imgs = config['train']['totoal_val_imgs']

train_images_path = config['train']['train_image_path']
train_mask_path = config['train']['train_mask_path']

val_images_path = config['train']['val_image_path']
val_mask_path = config['train']['val_mask_path']


def create_callbacks(saved_weights_name, tensorboard_logs):
    if not os.path.exists(tensorboard_logs):
        os.makedirs(tensorboard_logs)

    early_stop = EarlyStopping(
        monitor="loss",
        min_delta=0.01,
        patience=10,
        mode='min',
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=saved_weights_name,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        period=1,
        mode='min'
    )

    reduce_on_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='min',
        epsilon=0.01,
        cooldown=0,
        min_lr=0
    )

    tensor_board = TensorBoard(
        log_dir=tensorboard_logs,
        write_graph=True,
        write_images=True
    )

    return [early_stop, reduce_on_plateau, checkpoint, tensor_board]


callbacks = create_callbacks(config['train']['saved_weights'], config['train']['tensorboard_dir'])

train_images_gen = ImageDataGenerator(preprocessing_function=normalize)
train_images = train_images_gen.flow_from_directory(target_size=(512, 512),
                                                    class_mode=None,
                                                    directory=train_images_path,
                                                    seed=1,
                                                    batch_size=batch_size)
train_masks_gen = ImageDataGenerator(preprocessing_function=normalize)
train_masks = train_masks_gen.flow_from_directory(target_size=(512, 512),
                                                  class_mode=None,
                                                  color_mode='grayscale',
                                                  directory=train_mask_path,
                                                  seed=1,
                                                  batch_size=batch_size)

training_data = zip(train_images, train_masks)

val_images_gen = ImageDataGenerator(preprocessing_function=normalize)
val_images = val_images_gen.flow_from_directory(target_size=(512, 512),
                                                class_mode=None,
                                                directory=val_images_path,
                                                seed=1,
                                                batch_size=batch_size)
val_masks_gen = ImageDataGenerator(preprocessing_function=normalize)
val_masks = val_masks_gen.flow_from_directory(target_size=(512, 512),
                                              class_mode=None,
                                              color_mode='grayscale',
                                              directory=val_mask_path,
                                              seed=1,
                                              batch_size=batch_size)

val_data = zip(val_images, val_masks)
model = Seg().seg_network()
opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(opt, loss=loss)
model.fit_generator(training_data,
                    steps_per_epoch=total_training_imgs // batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=val_data,
                    validation_steps=total_val_imgs // batch_size
                    )
