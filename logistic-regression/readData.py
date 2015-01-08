"""Reads in the data from the text files."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import csv
import numpy as np

nfeatures = 2

def readFirst():
    x = np.empty([2,1])
    temp = np.empty([2,1])
    y = np.empty(1)
    with open('ex2data1.txt','r') as f:
        datafile = csv.reader(f)
        ii = 0
        for line in datafile:
            if ii==0:
                x[0,0] = line[0]
                x[1,0] = line[1]
                y[0] = line[2]
            temp[:,0] = line[:2]
            x=np.append(x,temp,axis=1)
            y=np.append(y,line[2])
            ii += 1
        # endfor line in datafile:
    # close file
    print x[0,0],x[1,0],y[0],x[0,-1],x[1,-1]
    print x.shape,y.shape
    return (x,y,ii)
