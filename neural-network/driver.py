"""Computes a neural network with backward propagation to determine weights."""

__author__="Jesse Lord"
__date__="January 12, 2015"

import readData
import costFunction
from sigmoid import sigmoidGradient
from randomInit import randomInit
from backPropagation import backPropagation
from predict import predict
from gradientChecking import gradientChecking

showcost = 0

if __name__=="__main__":
    input_layer = 400
    hidden_layer = 25
    num_labels = 10

    (X,y) = readData.readData()

    lam = 1

    if showcost:
        import numpy as np
        (ogTheta1,ogTheta2) = readData.readWeights()

        Thetas = np.reshape(ogTheta1,ogTheta1.size)
        Thetas = np.append(Thetas,ogTheta2)
        print costFunction.computeRegularizedCost(Thetas,
                                                  X,y,input_layer,hidden_layer,
                                                  num_labels,lam)

    Theta1 = randomInit(input_layer,hidden_layer)
    Theta2 = randomInit(hidden_layer,num_labels)

    gradientChecking(Theta1,Theta2,X,y,input_layer,hidden_layer,num_labels,lam)

    (Theta1,Theta2) = backPropagation(Theta1,Theta2,X,y,input_layer,
                                      hidden_layer,num_labels,lam)

    prediction = predict(X,y,Theta1,Theta2)
    print "Neural Network determines the handwriting correctly on the training set "+str(100*prediction)+"% of the time."
