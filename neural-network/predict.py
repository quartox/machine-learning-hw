"""Uses the neural network weights to compare how often the neural network correctly predicts the handwritten numbers."""

__author__="Jesse Lord"
__date__="January 12, 2015"

import numpy as np
from sigmoid import sigmoid

def predict(X,y,Theta1,Theta2):
    y = y[:,0]
    ndims = X.shape
    num_examples = ndims[0]
    m = ndims[1]

    # creating the input layer from the x-values

    #a1 = np.ones([m+1,num_examples])
    #for jj in range(m):
    #    a1[jj+1,:] = X[:,jj]
    #print a1.shape,X.shape
    #print np.amax(a1[:,2]),np.amin(a1[:,2])
    #print np.amax(X[:,1]),np.amin(X[:,1])

    a1 = np.ones([num_examples,m+1])
    a1[:,1:] = X
    a1 = np.transpose(a1)
    # computing the hidden layer values using the Theta1 matrix
    z2 = np.dot(Theta1,a1)

    debug = 0
    if debug:
        ndims = Theta1.shape
        print ndims,a1.shape

        z2debug = np.zeros([ndims[0],num_examples])

        for jj in range(ndims[0]):
            for kk in range(ndims[1]):
                z2debug[jj,:] += Theta1[jj,kk]*a1[kk,:]
        ndims = z2.shape
        print ndims,z2debug.shape
        maxdiff = 0.0
        for ii in range(ndims[0]):
            for jj in range(num_examples):
                if abs(z2[ii,jj]-z2debug[ii,jj]) > 0.0:
                    print "Here"
                if abs(z2[ii,jj]-z2debug[ii,jj]) > maxdiff:
                    maxdiff = abs(z2[ii,jj]-z2debug[ii,jj])
        print maxdiff
    ndims = z2.shape
    # creating the hidden layer outputs using the sigmoid function
    a2 = np.ones([ndims[0]+1,num_examples])
    a2[1:,:] = sigmoid(z2)
    # computing the output layer values using the Theta2 matrix
    z3 = np.dot(Theta2,a2)
    h = sigmoid(z3) # the output of the neural network

     # determining how well the neural network matches the training set
    num_labels = len(h[:,0])
    num_examples = len(h[0,:])

    ex = 1
    exmax = 0
    itermax = 0
    for ii in range(num_labels):
        if h[ii,ex] > exmax:
            exmax = h[ii,ex]
            itermax = ii

    #print y[ex],ii,h[ii,ex],np.amax(h[:,ex])
    #print h[:,ex]
    #exit()

    for ii in range(num_labels):
        if ii==0:
            hmax = h[ii,:]
            itermax = np.zeros([num_examples])+10
            diff = np.zeros([num_examples])
            absdiff = np.zeros([num_examples])
        else:
            for jj in range(num_examples):
                if h[ii,jj] > hmax[jj]:
                    hmax[jj] = h[ii,jj]
                    itermax[jj] = ii
                if ii == num_labels-1:
                    if jj == 0:
                        print z3[0,ex],h[0,ex],z3[9,ex],h[9,ex]
                        print itermax[ex],y[ex]
                        prediction = 0.0

                    diff[jj] = itermax[jj]-y[jj]
                    absdiff[jj] = abs(itermax[jj]-y[jj])
                    if int(itermax[jj]) == int(y[jj]):
                        prediction += 1.0/num_examples
                # endif h[jj] > hmax[jj]
            # endfor jj in range(len(h))
        # endif ii==0
    # endfor ii in range(num_labels)
    #print np.mean(diff),np.mean(absdiff)
    #print np.amin(diff),np.amax(diff)
    #print np.amin(absdiff),np.amax(absdiff)
    print np.amax(itermax),np.amin(itermax)
    return prediction
