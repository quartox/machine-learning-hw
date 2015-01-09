"""Reads in the data from the text files."""

__author__="Jesse Lord"
__date__="January 8, 2015"

import csv
import numpy as np

nfeatures = 2

def readFirst():
    x = np.empty([2,1])
    temp = np.empty([2,1])
    y = np.empty(1,dtype=float)
    with open('ex2data1.txt','r') as f:
        datafile = csv.reader(f)
        ii = 1
        for line in datafile:
            if ii==1:
                x[0,0] = line[0]
                x[1,0] = line[1]
                y[0] = np.float_(line[2])
            temp[:,0] = line[:2]
            x=np.append(x,temp,axis=1)
            y=np.append(y,np.float_(line[2]))
            ii += 1
        # endfor line in datafile:
    # close file
    return (x,y,ii)

def readSecond():
    x = np.empty([2,1])
    temp = np.empty([2,1])
    y = np.empty(1,dtype=float)
    with open('ex2data2.txt','r') as f:
        datafile = csv.reader(f)
        ii = 1
        for line in datafile:
            if ii==1:
                x[0,0] = line[0]
                x[1,0] = line[1]
                y[0] = np.float_(line[2])
            temp[:,0] = line[:2]
            x=np.append(x,temp,axis=1)
            y=np.append(y,np.float_(line[2]))
            ii += 1
        # endfor line in datafile:
    # close file
    return (x,y,ii)
