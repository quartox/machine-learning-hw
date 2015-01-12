"""Computes a neural network with predetermined weights."""

__author__="Jesse Lord"
__date__="January 12, 2015"

import readData
from predict import predict

if __name__=="__main__":
    input_layer = 400
    hidden_layer = 25
    num_labels = 10

    (X,y) = readData.readData()
    (Theta1,Theta2) = readData.readWeights()

    prediction = predict(X,y,Theta1,Theta2)
    prediction = np.mean(prediction==y)
    print "Neural Network determines the handwriting correctly on the training set "+str(100*prediction)+"% of the time."
