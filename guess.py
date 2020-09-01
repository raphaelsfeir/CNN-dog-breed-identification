#!/bin/python3
# -*- coding: utf-8 -*-
"""
@author: raphaël
"""
import sys, getopt
NB_BREEDS = 133
INPUT_SIZE = 512
BATCH_SIZE = 16

import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def main(argv):
    plot = False
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    try:
        opts,args = getopt.getopt(argv, "hi:p",["image="])
    except getopt.GetoptError:
        print("Syntax : guess.py -i <image> [-p]")
        sys.exit(2)
    for opt,arg in opts:
        if opt == '-h':
            print("Syntax : guess.py -i <image> [-p]")
            sys.exit()
        elif opt in ("-i", "--image"):
            img = arg
        elif opt in ("-p"):
            plot = True
   
    print("Image has been loaded!")
    print("Proceeding...")
    import tensorflow
    classifier = tensorflow.keras.models.load_model("models\\01-09-2020.0.82.hdf5")
    classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    
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
    
    import numpy as np
    from keras.preprocessing import image
    import matplotlib.pyplot as plt
    my_prediction = image.load_img(img, target_size=(512,512))
    my_prediction = image.img_to_array(my_prediction)
    my_prediction /= 255.
    my_prediction = np.expand_dims(my_prediction, axis=0)
    result = classifier.predict(my_prediction)
    breed = list(training_set.class_indices.keys())[np.argmax(result)].split(".")[1]
    
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        image = mpimg.imread(img)
        plt.axis("off")
        plt.imshow(image)
        plt.show()
    if np.max(result)*100 < 5:
        print("Is it really a dog?")
    else:
        print("{b} ({p:.2f}%)".format(b=breed, p=np.max(result)*100))
    sys.exit()
    
if __name__ == "__main__":
   main(sys.argv[1:])