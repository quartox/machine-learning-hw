"""Handwriting recognition using one vs all logistic regression."""

__author__="Jesse Lord"
__date__="January 9, 2015"

from readData import readData
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll
import pickle

firstrun = 0

if __name__=="__main__":
    (X,y) = readData()

    num_labels = 10 # number of labels for one vs all
    lam = 0.1 # regularization parameter

    if firstrun:
        all_thetas = oneVsAll(X,y,num_labels,lam)

        with open('thetas.pickle','w') as f:
            pickle.dump([all_thetas],f)
    else:
        with open('thetas.pickle') as f:
            [all_thetas] = pickle.load(f)

    prediction = predictOneVsAll(X,y,all_thetas,num_labels)
    print "One vs All determines the handwriting correctly on the training set "+str(100*prediction)+"% of the time."
