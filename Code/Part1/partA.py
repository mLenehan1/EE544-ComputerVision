%tensorflow_version 2.x

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import constraints
from tensorflow.keras import regularizers
from google.colab import drive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)
print(tf.test.gpu_device_name())

drive.mount('/content/drive')
root_dir = "/content/drive/My\ Drive/"
base_dir = root_dir + 'Colab\ Notebooks/EE544/Assignment/'
print('Extracting Data')
!unzip -qo {base_dir + 'imagenette_6class.zip'} 
print("Extraction Done!")

trainDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    zoom_range=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1./255
)

validDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    zoom_range=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1./255
)

testDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    zoom_range=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1./255
)

trainGenerator = trainDataGenerator.flow_from_directory(
    directory=r"./train/",
    target_size=(224,224),
    batch_size=50,
    class_mode="categorical",
)

validGenerator = validDataGenerator.flow_from_directory(
    directory=r"./validation/",
    target_size=(224,224),
    batch_size=50,
    class_mode="categorical",
)

testGenerator = testDataGenerator.flow_from_directory(
    directory=r"./test/",
    target_size=(224,224),
    batch_size=1,
    class_mode=None,
    shuffle=False
)

model = tf.keras.Sequential()
model.add(layers.Conv2D(
    filters=32, kernel_size=(3,3), input_shape=(224,224,3),
    strides=(1,1), padding='same', 
    dilation_rate=(1,1), activation='relu',
    kernel_initializer='glorot_uniform',
    kernel_regularizer=regularizers.l2(0.01)
))
model.add(layers.Conv2D(
    filters=32, kernel_size=(3,3), strides=(1,1), padding='same', 
    dilation_rate=(1,1), activation='relu',
    kernel_initializer='glorot_uniform',
    kernel_regularizer=regularizers.l2(0.01)
))
model.add(layers.MaxPooling2D(
    pool_size=(2,2)
))
model.add(layers.Conv2D(
    filters=64, kernel_size=(3,3), strides=(1,1), padding='same', 
    dilation_rate=(1,1), activation='relu',
    kernel_initializer='glorot_uniform',
    kernel_regularizer=regularizers.l2(0.01)
))
model.add(layers.Conv2D(
    filters=64, kernel_size=(3,3), strides=(1,1), padding='same', 
    dilation_rate=(1,1), activation='relu',
    kernel_initializer='glorot_uniform',
    kernel_regularizer=regularizers.l2(0.01)
))
model.add(layers.MaxPooling2D(
    pool_size=(2,2), strides=None
))
model.add(layers.Flatten())
model.add(layers.Dense(
    units=512,
    activation='relu',
    kernel_initializer='glorot_uniform',
    kernel_regularizer=regularizers.l2(1e-5)
))
model.add(layers.Dense(
    units=6,
    activation='softmax',
    kernel_initializer='glorot_uniform'
))

model.compile(
    optimizer=optimizers.Adam(), 
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

trainSteps = trainGenerator.n//trainGenerator.batch_size
validSteps = validGenerator.n//validGenerator.batch_size
testSteps = testGenerator.n//testGenerator.batch_size

history = model.fit(
    x=trainGenerator,
    steps_per_epoch=trainSteps,
    validation_data=validGenerator,
    validation_steps=validSteps,
    epochs=30
)

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

scores = model.evaluate(
    x=validGenerator,
    steps=validSteps,
)

print('CNN Accuracy: %.2f%%' % (scores[1]*100.0))
print('CNN Error: %.2f%%' % (100-scores[1]*100))

drive.flush_and_unmount()
print('All changes made in this colab session should now be visible in Drive.')