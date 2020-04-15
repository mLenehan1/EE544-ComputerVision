%tensorflow_version 2.x

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import constraints
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
from google.colab import drive
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import time

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
    kernel_regularizer=None
))
model.add(layers.Conv2D(
    filters=32, kernel_size=(3,3), strides=(1,1), padding='same', 
    dilation_rate=(1,1), activation='relu',
    kernel_initializer='glorot_uniform',
    kernel_regularizer=None
))
model.add(layers.BatchNormalization(axis=1))
model.add(layers.MaxPooling2D(
    pool_size=(2,2)
))
model.add(layers.Conv2D(
    filters=64, kernel_size=(3,3), strides=(1,1), padding='same', 
    dilation_rate=(1,1), activation='relu',
    kernel_initializer='glorot_uniform',
    kernel_regularizer=None
))
model.add(layers.Conv2D(
    filters=64, kernel_size=(3,3), strides=(1,1), padding='same', 
    dilation_rate=(1,1), activation='relu',
    kernel_initializer='glorot_uniform',
    kernel_regularizer=None
))
model.add(layers.BatchNormalization(axis=1))
model.add(layers.MaxPooling2D(
    pool_size=(2,2), strides=None
))
model.add(layers.Flatten())
model.add(layers.Dense(
    units=512,
    activation='relu',
    kernel_initializer='glorot_uniform',
    kernel_regularizer=regularizers.l2(0.00025)
))
model.add(layers.BatchNormalization(axis=1))
model.add(layers.Dense(
    units=6,
    activation='softmax',
    kernel_initializer='glorot_uniform'
))

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001), 
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)

model.save('/content/drive/My Drive/Colab Notebooks/EE544/Assignment/q1pd.h5')


print(model.summary())

trainSteps = trainGenerator.n//trainGenerator.batch_size
validSteps = validGenerator.n//validGenerator.batch_size
testSteps = testGenerator.n//testGenerator.batch_size

es = callbacks.EarlyStopping(
    monitor='val_loss', patience=5, mode='auto'
)

trainTimeStart = time.perf_counter()

history = model.fit(
    x=trainGenerator,
    steps_per_epoch=trainSteps,
    validation_data=validGenerator,
    validation_steps=validSteps,
    epochs=30,
    callbacks = [es]
)

trainTimeEnd = time.perf_counter()

totalTime = trainTimeEnd - trainTimeStart

print("Total Time = %.4f" % totalTime)

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

testGenerator.reset()
pred = model.predict(testGenerator, steps=testSteps, verbose=1)
predictedClass = np.argmax(pred, axis=-1)
predictedClassIndices = np.argmax(pred, axis=1)
labels = (testGenerator.labels)
classLabels = (testGenerator.classes)

targetNames = ['Church', 'English Springer', 'Garbage Truck', 'Gas Pump',
               'Parachute', 'Tench']
print(classification_report(predictedClassIndices, labels, target_names=targetNames))

matrix = confusion_matrix(predictedClassIndices, labels)
print(matrix)
seaborn.set(font_scale=1.4)
seaborn.heatmap(matrix, annot=True, annot_kws={"size": 16})
plt.show()

drive.flush_and_unmount()
print('All changes made in this colab session should now be visible in Drive.')