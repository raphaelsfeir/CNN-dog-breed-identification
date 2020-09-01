# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 22:25:07 2020

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


def build_cnn(loaded_model=None):
    if loaded_model:
        # Loading model
        classifier = tensorflow.keras.models.load_model(loaded_model)
    else:
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
    return classifier

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


# Checkpoints (used on Google Colab)
from keras.callbacks import EarlyStopping, ModelCheckpoint
callback = EarlyStopping(monitor='loss', patience=3)
checkpoint = ModelCheckpoint("/content/drive/My Drive/Perso/IA/CNN-for-dog-breed/checkpoint-bis-{epoch:02d}-{val_accuracy:.2f}.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


# GO!
history = classifier.fit(
        training_set,
        steps_per_epoch=NB_TRAIN // BATCH_SIZE,
        epochs=NB_EPOCH,
        validation_data=test_set,
        validation_steps=NB_VALID // BATCH_SIZE,
        callbacks=[callback, checkpoint])

# Final saving
classifier.save("/content/drive/My Drive/Perso/IA/CNN-for-dog-breed/final-xception.h5")