"""
author: alan cabrera
references:
1.https://arxiv.org/pdf/1505.04597.pdf
2.https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
3.https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5

   The convolutional neural network, or CNN,
is a kind of neural network model designed 
to working with two-dimensional image data.
   It make use of a convolutional layer that 
gives the network its name. This layer 
performs an operation called a convolution 
which is essentially taking the dot product
using a set of weights, or filters, and an
array derived from an input image.
    
tensorflow is a machine learning library
keras is a neaural network library
"""
import tensorflow as tf #keras is integrated into tensorflow, as noted from NuSeT

#For importing the dataset (thanks Brendan!)
from cell_counter.import_dataset import load_synthetic_dataset


