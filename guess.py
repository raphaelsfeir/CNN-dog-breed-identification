#!/bin/python3
# -*- coding: utf-8 -*-
"""
@author: raphaël
"""
import sys, getopt

# constants
NB_BREEDS = 133
INPUT_SIZE = 512

# remove the Tensorflow messages
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def main(argv):
    plot = False
    img = ''
    
    # Fix for the Truncated error
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # Parsing the arguments
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


    # Uploading image
    try:
        import numpy as np
        from keras.preprocessing import image
        my_prediction = image.load_img(img, target_size=(512,512))
        my_prediction = image.img_to_array(my_prediction)
        my_prediction /= 255.
        my_prediction = np.expand_dims(my_prediction, axis=0)
        print("Image has been loaded!")
    except:
        print("No image found!")
        print("Syntax : guess.py -i <image> [-p]")
        sys.exit(2)
    print("Processing...")
    
    # Uploading model
    import tensorflow
    try:
        classifier = tensorflow.keras.models.load_model("models\\model.hdf5")
    except:
        print("No model found!")
        import requests
        with open("models\\model.hdf5", "wb") as f:
            print("Downloading...")
            r = requests.get("https://raphaelsfeir.com/FileManager/e042ce55/latest.hdf5", stream=True)
            length = r.headers.get("Content-Length")
            if length is None:
                f.write(r.content)
            else:
                dl = 0
                length = int(length)
                for data in r.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / length)
                    sys.stdout.write("\r[%s%s] %s" % ('=' * done, ' ' * (50-done), str(done*2) + "%") )    
                    sys.stdout.flush()
        print("The model has been downloaded successfully!")
        classifier = tensorflow.keras.models.load_model("models\\model.hdf5")
    
    # Compiling the model
    classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    
    # Making the prediction
    breeds={'001.Affenpinscher':0,'002.Afghan hound':1,'003.Airedale terrier':2,'004.Akita':3,'005.Alaskan malamute':4,'006.American eskimo dog':5,'007.American foxhound':6,'008.American staffordshire terrier':7,'009.American water spaniel':8,'010.Anatolian shepherd dog':9,'011.Australian cattle dog':10,'012.Australian shepherd':11,'013.Australian terrier':12,'014.Basenji':13,'015.Basset hound':14,'016.Beagle':15,'017.Bearded collie':16,'018.Beauceron':17,'019.Bedlington terrier':18,'020.Belgian malinois':19,'021.Belgian sheepdog':20,'022.Belgian tervuren':21,'023.Bernese mountain dog':22,'024.Bichon frise':23,'025.Black and tan coonhound':24,'026.Black russian terrier':25,'027.Bloodhound':26,'028.Bluetick coonhound':27,'029.Border collie':28,'030.Border terrier':29,'031.Borzoi':30,'032.Boston terrier':31,'033.Bouvier des flandres':32,'034.Boxer':33,'035.Boykin spaniel':34,'036.Briard':35,'037.Brittany':36,'038.Brussels griffon':37,'039.Bull terrier':38,'040.Bulldog':39,'041.Bullmastiff':40,'042.Cairn terrier':41,'043.Canaan dog':42,'044.Cane corso':43,'045.Cardigan welsh corgi':44,'046.Cavalier king charles spaniel':45,'047.Chesapeake bay retriever':46,'048.Chihuahua':47,'049.Chinese crested':48,'050.Chinese shar-pei':49,'051.Chow chow':50,'052.Clumber spaniel':51,'053.Cocker spaniel':52,'054.Collie':53,'055.Curly-coated retriever':54,'056.Dachshund':55,'057.Dalmatian':56,'058.Dandie dinmont terrier':57,'059.Doberman pinscher':58,'060.Dogue de bordeaux':59,'061.English cocker spaniel':60,'062.English setter':61,'063.English springer spaniel':62,'064.English toy spaniel':63,'065.Entlebucher mountain dog':64,'066.Field spaniel':65,'067.Finnish spitz':66,'068.Flat-coated retriever':67,'069.French bulldog':68,'070.German pinscher':69,'071.German shepherd dog':70,'072.German shorthaired pointer':71,'073.German wirehaired pointer':72,'074.Giant schnauzer':73,'075.Glen of imaal terrier':74,'076.Golden retriever':75,'077.Gordon setter':76,'078.Great dane':77,'079.Great pyrenees':78,'080.Greater swiss mountain dog':79,'081.Greyhound':80,'082.Havanese':81,'083.Ibizan hound':82,'084.Icelandic sheepdog':83,'085.Irish red and white setter':84,'086.Irish setter':85,'087.Irish terrier':86,'088.Irish water spaniel':87,'089.Irish wolfhound':88,'090.Italian greyhound':89,'091.Japanese chin':90,'092.Keeshond':91,'093.Kerry blue terrier':92,'094.Komondor':93,'095.Kuvasz':94,'096.Labrador retriever':95,'097.Lakeland terrier':96,'098.Leonberger':97,'099.Lhasa apso':98,'100.Lowchen':99,'101.Maltese':100,'102.Manchester terrier':101,'103.Mastiff':102,'104.Miniature schnauzer':103,'105.Neapolitan mastiff':104,'106.Newfoundland':105,'107.Norfolk terrier':106,'108.Norwegian buhund':107,'109.Norwegian elkhound':108,'110.Norwegian lundehund':109,'111.Norwich terrier':110,'112.Nova scotia duck tolling retriever':111,'113.Old english sheepdog':112,'114.Otterhound':113,'115.Papillon':114,'116.Parson russell terrier':115,'117.Pekingese':116,'118.Pembroke welsh corgi':117,'119.Petit basset griffon vendeen':118,'120.Pharaoh hound':119,'121.Plott':120,'122.Pointer':121,'123.Pomeranian':122,'124.Poodle':123,'125.Portuguese water dog':124,'126.Saint bernard':125,'127.Silky terrier':126,'128.Smooth fox terrier':127,'129.Tibetan mastiff':128,'130.Welsh springer spaniel':129,'131.Wirehaired pointing griffon':130,'132.Xoloitzcuintli':131,'133.Yorkshire terrier':132}    
    result = classifier.predict(my_prediction)
    breed = list(breeds.keys())[np.argmax(result)].split(".")[1]
    
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        image = mpimg.imread(img)
        plt.axis("off")
        plt.imshow(image)
        plt.show()
    
    # Printing the result
    if np.max(result)*100 < 5:
        print("Is it really a dog?")
    else:
        print("{b} ({p:.2f}%)".format(b=breed, p=np.max(result)*100))
    sys.exit()
    
if __name__ == "__main__":
   main(sys.argv[1:])