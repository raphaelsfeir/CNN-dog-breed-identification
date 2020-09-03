<p align="center">
  <img src="https://raphaelsfeir.com/FileManager/e042ce55/logo.png" alt="logo" width="150px"/>
</p>

# Dog Breed Identification (CNN)
This is my first real AI project. The purpose is to identify a dog's breed thanks to a photo using a CNN. First I built the CNN from scratch but because of the huge dataset, I quickly decided to use transfer learning because of the speed and the accuracy.

<p align="center">
  <img src="https://github.com/raphaelsfeir/CNN-dog-breed-identification/blob/master/jackson_best.png" alt="Here is my dog!" />
</p>

## How does the model work?
1) The input image is resized to a 512x512 image
2) This image is firstly processed through the [XCeption](https://arxiv.org/abs/1610.02357) network
3) Then comes a sequential model

![model](https://github.com/raphaelsfeir/CNN-dog-breed-identification/blob/master/model.png "Model")

### Summary
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
xception (Functional)        (None, 2048)              20861480  
_________________________________________________________________
batch_normalization_15 (Batc (None, 2048)              8192      
_________________________________________________________________
dropout_16 (Dropout)         (None, 2048)              0         
_________________________________________________________________
dense_16 (Dense)             (None, 1024)              2098176   
_________________________________________________________________
dropout_17 (Dropout)         (None, 1024)              0         
_________________________________________________________________
dense_17 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dropout_18 (Dropout)         (None, 1024)              0         
_________________________________________________________________
dense_18 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dropout_19 (Dropout)         (None, 1024)              0         
_________________________________________________________________
dense_19 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dropout_20 (Dropout)         (None, 1024)              0         
_________________________________________________________________
dense_20 (Dense)             (None, 1024)              1049600   
_________________________________________________________________
dropout_21 (Dropout)         (None, 1024)              0         
_________________________________________________________________
dense_21 (Dense)             (None, 133)               136325    
=================================================================
Total params: 27,302,573
Trainable params: 6,436,997
Non-trainable params: 20,865,576
_________________________________________________________________
```

## Running the program
I used the Jupyter Notebook (ready for Google Colab!) for the main development. Then, for some technical issues I create the normal Python file: you have the choice!
One little note though :
- Python version : 3.7.7
- Tensorflow version : 2.3.0
- Download the dataset [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
- You can download the functionnal model [here](https://raphaelsfeir.com/FileManager/e042ce55/latest.hdf5) (`.hdf5` file)

## How can I test it with my own photos?
Just run this code in your terminal :)
```bash
python guess.py -i <your image> [-p]
```
The `-p` flag will show the image you gave to the network.