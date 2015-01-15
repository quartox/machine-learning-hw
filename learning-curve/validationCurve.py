"""Computes the validation curve for a range of regularization values."""

__author__="Jesse Lord"
__date__="January 15, 2015"

def validationCurve(theta,X,y,Xcv,ycv):
    lam = np.zeros(1)
    ii=0.001
    while ii<=10:
        lam = np.append(lam,ii)
        ii *= 3.0
    error_train = np.empty(lam.size)
    error_cv = np.empty(lam.size)
    for ii in range(lam.size):
        theta_lam = regression(theta,X,y,lam[ii])
        # set lambda=0 to compute error in training and cross-validation sets
        error_train[ii] = computeCost(theta_lam,X,y,0)
        error_cv[ii] = computeCost(theta_lam,Xcv,ycv,0)
    return (error_train,error_cv,lam)
