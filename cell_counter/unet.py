#author: alan cabrera
#
#resources:https://arxiv.org/pdf/1505.04597.pdf
#
#tensorflow is a machine learning library
#keras is a neaural network library

import tensorflow as tf #keras is integrated into tensorflow, as noted from NuSeT

#the model will be built in a u-shape,where the left side 
#contracts and the right side expands. 

def build_model():