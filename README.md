# Dog Breed Identification (CNN)
This is my first real AI project. The purpose is to identify a dog's breed thanks to a photo using a CNN. First I built the CNN from scratch but because of the huge dataset, I quickly decided to use transfer learning because of the speed and the accuracy.

![Running the program with my dog](https://github.com/raphaelsfeir/CNN-dog-breed-identification/blob/master/jackson_best.png "Running the program with my dog")

## How does the model work?
1) The network takes a 512x512x3 image as an input.
2) This image is given to the [XCeption](https://arxiv.org/abs/1610.02357) network
3) Then comes a sequential model

![model](https://github.com/raphaelsfeir/CNN-dog-breed-identification/blob/master/model.png "Model")

## Running the program
I used the Jupyter Notebook (ready for Google Colab!) for the main development. Then, for some technical issues I create the normal Python file: you have the choice!
One little note though :
- Python version : 3.7.7
- Tensorflow version : 2.3.0
- Download the dataset [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

## How can I test it for my own photos?
Just run this code in your terminal :)
```bash
python guess.py -i <your image> [-p]
```
The `-p` flag will show the image you gave to the network.
