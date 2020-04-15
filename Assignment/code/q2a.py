%tensorflow_version 2.x

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import constraints
from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
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
!unzip -qo {base_dir + 'food101_4class.zip'} 
print("Extraction Done!")

trainDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = tf.keras.applications.resnet50.preprocess_input
)

validDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator()

testDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator()

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

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3)
)

x = base_model.output
x = layers.Flatten()(x)
preds = layers.Dense(4, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=preds)

for layer in model.layers[:143]:
  layer.trainable = False

for layer in model.layers[143:]:
  layer.trainable = True

trainSteps = trainGenerator.n//trainGenerator.batch_size
validSteps = validGenerator.n//validGenerator.batch_size
testSteps = testGenerator.n//testGenerator.batch_size


model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.save('/content/drive/My Drive/Colab Notebooks/EE544/Assignment/q2pa.h5')

print(model.summary())

trainTimeStart = time.perf_counter()

history = model.fit(
    trainGenerator, 
    steps_per_epoch=trainSteps, 
    epochs = 5,
    validation_data = validGenerator,
    validation_steps = validSteps,
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

targetNames = ['Chicken Curry', 'Hamburger', 'Omelette', 'Waffles']
print(classification_report(predictedClassIndices, labels, target_names=targetNames))

matrix = confusion_matrix(predictedClassIndices, labels)
print(matrix)
seaborn.set(font_scale=1.4)
seaborn.heatmap(matrix, annot=True, annot_kws={"size": 16})
plt.show()

drive.flush_and_unmount()
print('All changes made in this colab session should now be visible in Drive.')