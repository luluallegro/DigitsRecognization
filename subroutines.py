'''
#  ======================================================================
#     FileName: subroutines.py
#      Project: HandDigitRecognization
#       Author: Yong Yang, Department of Mathematics, UTA
#        Email: yong.yang@mavs.uta.edu
#      Created: 2017-02-20 11:26:34
#   LastChange: 2017-02-20 21:40:23
#  ======================================================================
'''
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

def displayData(X,example_width=None):

    # Compute rows, cols
    m,n=X.shape

    if example_width == None:
        example_width=round(math.sqrt(n))

    example_height=n//example_width

    # Compute number of items to display
    display_rows=math.floor(math.sqrt(m))
    display_cols=math.ceil(m/display_rows)

    # Between images padding
    pad=1

    # Set up blank display
    display_array=-np.ones((pad + display_rows * (example_height + pad),
        pad + display_cols * (example_width + pad)))
    # Copy each example into a patch on the display array
    curr_ex=0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex==m: break
            # Get the max value of the patch
            max_val=abs(X[curr_ex,:]).max()
            height_start=pad+j*(example_height+pad)
            width_start=pad+i*(example_width+pad)
            display_array[height_start:height_start+example_height,
                    width_start:width_start+example_width]=np.reshape(X[curr_ex,:],(example_height,example_width)).T/max_val
            curr_ex+=1

        if curr_ex==m: break

    imgplt=plt.imshow(display_array,cmap='gray',vmin=-1,vmax=1)
    plt.axis('off')
    plt.show()
    return

def oneVsAll(X,y,num_labels,lamd):
    m,n=X.shape
    all_theta=np.zeros((num_labels,n+1))
    X=np.insert(X,0,1,axis=1)
    for k in range(num_labels):
        initial_theta=np.zeros((1,n+1))
        all_theta[k,:]=optimize.fmin_cg(lrJ,initial_theta,lrJprime,(X,(y==k)*1.0,lamd),disp=False,gtol=1e-5,maxiter=200)

    return all_theta

def lrJ(theta,X,y,lamd):
    m=y.shape[0]
    J=0
    htheta=sigmoid(X.dot(theta))
    temp=theta
    temp[0]=0
    J=(-np.log(htheta).dot(y)-np.log(1-htheta).dot(1-y))/m+sum(temp*temp)*lamd/(2*m)
    return J

def lrJprime(theta,X,y,lamd):
    m=y.shape[0]
    theta=theta.reshape((theta.shape[0],1))
    htheta=sigmoid(X.dot(theta))
    temp=theta
    temp[0]=0
    #  print(theta.shape)
    #  input()
    grad=X.T.dot(htheta-y)/m+lamd/m*temp
    return grad.reshape((grad.shape[0],))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def predictOneVsAll(all_theta,X):
    m=X.shape[0]
    num_labels=all_theta.shape[0]
    p=np.zeros((m,1))
    X=np.insert(X,0,1,axis=1)

    MP=np.dot(X,all_theta.T)
    p=np.argmax(MP,axis=1)
    return p

