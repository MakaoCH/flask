import pandas as pd
import numpy as np

import os
import cv2
from PIL import Image

from tqdm.auto import trange, tqdm

import keras
import tensorflow as tf
from tqdm.keras import TqdmCallback

from matplotlib import pyplot as plt
from ipywidgets import interact

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from keras.preprocessing.image import ImageDataGenerator


train = pd.read_csv('data/train.csv')
test = np.genfromtxt('data/test.csv', delimiter=',', skip_header=1)
sample_submission = pd.read_csv('data/sample_submission.csv')

train.shape, test.shape, sample_submission.shape

labels = train['label']
train.drop(['label'], axis=1, inplace=True)

train = np.array(train, dtype=float)

train = train.reshape(42000, 28, 28)
test = test.reshape(28000, 28, 28)

img_rows, img_cols = (32, 32)
input_shape = (img_rows, img_cols, 3)

def resizeImage(images, new_size):
    nimages = np.zeros((images.shape[0], new_size[0], new_size[1], 3))
    for image in trange(images.shape[0]):
        nimages[image, :new_size[0], :new_size[1], 0] = cv2.resize(images[image], new_size)
        nimages[image, :new_size[0], :new_size[1], 1] = cv2.resize(images[image], new_size)
        nimages[image, :new_size[0], :new_size[1], 2] = cv2.resize(images[image], new_size)
    return nimages

train = resizeImage(train, (img_rows, img_cols))
test = resizeImage(test, (img_rows, img_cols))

val_split = 0.05
BATCH_SIZE = 256
TRAIN_STEPS_PER_EPOCH = train.shape[0]*(1-val_split)//BATCH_SIZE
VAL_STEPS_PER_EPOCH = train.shape[0]*val_split//BATCH_SIZE

train_datagen = ImageDataGenerator(rescale=1/255.0,
                                   rotation_range=15,
                                   zoom_range=0.15,
                                   validation_split=val_split,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   shear_range=0.15,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1/255.0)

val_split = 0.05
BATCH_SIZE = 256
TRAIN_STEPS_PER_EPOCH = train.shape[0]*(1-val_split)//BATCH_SIZE
VAL_STEPS_PER_EPOCH = train.shape[0]*val_split//BATCH_SIZE

train_aug = train_datagen.flow(train,
                               labels,
                               batch_size=BATCH_SIZE,
                               subset='training',
                               shuffle=False,
                               seed=42)

valid_aug = train_datagen.flow(train,
                               labels,
                               batch_size=BATCH_SIZE,
                               subset='validation',
                               shuffle=False,
                               seed=42)


test_aug = train_datagen.flow(test,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              seed=42)

def build_model():
    cnn = keras.models.Sequential()

    cnn.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='random_uniform', input_shape=input_shape))
    cnn.add(keras.layers.Dropout(0.5))
    cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

    cnn.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='random_uniform'))
    cnn.add(keras.layers.Dropout(0.5))
    cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

    cnn.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='random_uniform'))
    cnn.add(keras.layers.Dropout(0.5))
    cnn.add(keras.layers.MaxPool2D(pool_size=(3, 3)))

    cnn.add(keras.layers.Flatten())
    cnn.add(keras.layers.Dense(units=128, activation='relu'))

    cnn.add(keras.layers.Dense(units=10, activation='softmax'))
    
    cnn.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return cnn

cnn = build_model()

#keras.utils.plot_model(cnn, show_shapes=True, show_layer_names=True)

ch = tf.keras.callbacks.ModelCheckpoint(
    filepath='mnist_model.h5',
    save_weights_only=False,
    monitor='acc',
    mode='max',
    save_best_only=True
)

es = tf.keras.callbacks.EarlyStopping(
    monitor='acc',
    min_delta=0.003,
    patience=15,
    mode='max',
    restore_best_weights=True,
)

lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='acc',
    factor=0.05,
    patience=3,
    mode='max',
)

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

history = cnn.fit(train_aug, validation_data=valid_aug, epochs=50, verbose=0,
                  callbacks=[ch, es, lr, TqdmCallback(verbose=1)])

y_pred = np.argmax(cnn.predict(test), axis=-1)


sample_submission['Label'] = y_pred
sample_submission.to_csv('ss.csv', index=False)

cnn.save('mnist_model.h5')

