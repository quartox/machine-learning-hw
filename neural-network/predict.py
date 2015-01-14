"""Uses the neural network weights to compare how often the neural network correctly predicts the handwritten numbers."""

__author__="Jesse Lord"
__date__="January 12, 2015"

debug = 0

import numpy as np
from sigmoid import sigmoid
if debug:
    import matplotlib.pyplot as p

def predict(X,y,Theta1,Theta2):
    y = y[:,0]
    ndims = X.shape
    num_examples = ndims[0]
    num_pixels = ndims[1]

    # creating the input layer from the x-values
    a1 = np.ones([num_examples,num_pixels+1])
    a1[:,1:] = X

    # computing the hidden layer values using the Theta1 matrix
    z2 = np.dot(a1,np.transpose(Theta1))
    ndims = z2.shape

    # creating the hidden layer outputs using the sigmoid function
    a2 = np.ones([num_examples,ndims[1]+1])
    a2[:,1:] = sigmoid(z2)

    # computing the output layer values using the Theta2 matrix
    z3 = np.dot(a2,np.transpose(Theta2))
    h = sigmoid(z3) # the output of the neural network

    # determining how well the neural network matches the training set
    # the indexing is off by one since the data set comes from octave
    # which is a language with indexing that starts at 1
    prediction = np.argmax(h,axis=1)+1

    if debug:
        ex = 500
        print y[ex],prediction[ex]
        a = X[ex,:]
        a = a.reshape([20,20])
        p.imshow(a)
        p.show()

    prediction = np.mean(prediction==y)

    return prediction
