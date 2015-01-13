"""Reads in the data from the matlab files."""

__author__="Jesse Lord"
__date__="January 12, 2015"
import scipy.io as io

def readData():
    mat = io.loadmat("ex4data1.mat")
    y = mat['y']
    X = mat['X']
    return (X,y)

def readWeights():
    mat = io.loadmat("ex4weights.mat")
    Theta1 = mat['Theta1']
    Theta2 = mat['Theta2']
    return (Theta1,Theta2)
