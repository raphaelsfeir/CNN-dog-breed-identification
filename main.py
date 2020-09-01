# -*- coding: utf-8 -*-
"""
@author: raphaël
"""

# Fix for Truncated File OS Error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# CONSTANTS
NB_BREEDS = 133
INPUT_SIZE = 512
BATCH_SIZE = 16
NB_TRAIN = 6680
NB_VALID = 835
NB_EPOCH = 20

# 1 - CNN construction
from keras.models import Sequential
from keras.models import load_model
import tensorflow

# Loading model
classifier = tensorflow.keras.models.load_model("models\\01-09-2020.0.82.hdf5")
classifier.summary()

from keras.applications import Xception

# CNN initialization
classifier = Sequential()

xception = Xception(weights='imagenet', 
                                 include_top = False, 
                                 input_shape = (INPUT_SIZE, INPUT_SIZE,3),
                                 pooling = 'max')
classifier.add(xception)
# Freezing the XCeption layers
for layer in classifier.layers:
    layer.trainable = False

from keras.layers import Dense, Dropout, BatchNormalization
classifier.add(BatchNormalization())

# Fully connected
classifier.add(Dropout(0.1))
classifier.add(Dense(units=1024, activation="relu"))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=1024, activation="relu"))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=1024, activation="relu"))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=1024, activation="relu"))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=1024, activation="relu"))
classifier.add(Dropout(0.1))

# Predictions
classifier.add(Dense(units=NB_BREEDS, activation="softmax"))

# Compilation
from keras.optimizers import RMSprop
classifier.summary()
classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

# Let's train!
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'dataset\\train',
        target_size=(INPUT_SIZE,INPUT_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        'dataset\\valid',
        target_size=(INPUT_SIZE,INPUT_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')


# Checkpoints
from keras.callbacks import EarlyStopping, ModelCheckpoint
callback = EarlyStopping(monitor='loss', patience=3)
checkpoint = ModelCheckpoint("<YOUR_PATH>", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


# GO!
history = classifier.fit(
        training_set,
        steps_per_epoch=NB_TRAIN // BATCH_SIZE,
        epochs=NB_EPOCH,
        validation_data=test_set,
        validation_steps=NB_VALID // BATCH_SIZE,
        callbacks=[callback, checkpoint])

# Final saving
classifier.save("<YOUR_PATH>")


# For single predictions :)
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

IMAGE = "single_predictions\\error.jpg" # Just put the path to your image
my_prediction = image.load_img(IMAGE, target_size=(512,512))
my_prediction = image.img_to_array(my_prediction)
my_prediction /= 255.
my_prediction = np.expand_dims(my_prediction, axis=0)
result = classifier.predict(my_prediction)


# Some stuff to build a nice result
breed = list(training_set.class_indices.keys())[np.argmax(result)].split(".")[1]

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread(IMAGE)
plt.axis("off")
plt.imshow(image)
plt.show()
if np.max(result)*100 < 5: # If all the predictions are <5% we can consider the subject isn't a dog
    print("Is it really a dog?")
else:
    print("{b} ({p:.2f}%)".format(b=breed, p=np.max(result)*100))