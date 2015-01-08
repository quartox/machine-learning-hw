"""Reads the machine learning homework data set."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import csv
import numpy as np

def readSingleFeature():
    nexamples = 97 # learned from checking the file size
    x = np.empty([nexamples],dtype=float)
    y = np.empty([nexamples],dtype=float)
    with open('ex1data1.txt','r') as f:
        datafile = csv.reader(f)
        ii = 0
        for line in datafile:
            x[ii] = line[0]
            y[ii] = line[1]
            ii += 1

    #plot.plot(x,y,'o')
    #plot.show()
    return (x,y)

def readMultiFeature():
    nexamples = 47 # learned from checking the file size
    x = np.empty([2,nexamples],dtype=float)
    y = np.empty([nexamples],dtype=float)
    with open('ex1data2.txt','r') as f:
        datafile = csv.reader(f)
        ii = 0
        for line in datafile:
            x[0,ii] = line[0]
            x[1,ii] = line[1]
            y[ii] = line[2]
            ii += 1

    #plot.plot(x,y,'o')
    #plot.show()
    return (x,y,nexamples)
