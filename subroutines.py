'''
#  ======================================================================
#     FileName: subroutines.py
#      Project: DigitsRecognization
#       Author: Yong Yang, Department of Mathematics, UTA
#        Email: yongyang@uta.edu; yongyang.math@gmail.com
#      Created: 2017-02-20 11:26:34
#   LastChange: 2017-02-24 11:57:33
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

def sigmoidGradient(z):
    gz=sigmoid(z)
    return np.multiply(gz,(1-gz))

def predictOneVsAll(all_theta,X):
    m=X.shape[0]
    num_labels=all_theta.shape[0]
    p=np.zeros((m,1))
    X=np.insert(X,0,1,axis=1)

    MP=np.dot(X,all_theta.T)
    p=np.argmax(MP,axis=1)
    return p

def randInitializeWeights(L_in,L_out):
    eps_init=0.12
    return np.random.rand(L_out,L_in+1)*2*eps_init-eps_init

def debugInitializeWeights(fan_out,fan_in):
    fan_dim=fan_out*(fan_in+1)
    return np.sin(np.linspace(1,fan_dim,fan_dim)).reshape(fan_out,fan_in+1)/10

def checkNNGradients(lamd):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    X=debugInitializeWeights(m,input_layer_size-1)
    y=1+np.mod(np.linspace(1,m,m),num_labels)

    nn_params=np.r_[Theta1.T.ravel().T,Theta2.T.ravel().T]
    cost,grad=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamd)
    options=(input_layer_size,hidden_layer_size,num_labels,X,y,lamd)
    numgrad=computeNumerialGradient(options,nn_params)
    print(np.concatenate((numgrad,grad),axis=1))
    print('The above two columns you get should be very similar.\n'+
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')
    return np.linalg.norm(numgrad-grad,ord=2)/np.linalg.norm(numgrad+grad,ord=2)




def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamd):
    Theta1 = np.matrix(np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1),order='F'))
    Theta2 = np.matrix(np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1),order='F'))
    m=X.shape[0]
    J=0.0
    Theta1_grad=np.zeros(Theta1.shape)
    Theta2_grad=np.zeros(Theta2.shape)

    X=np.insert(X,0,1,axis=1)
    X=np.asmatrix(X)
    Z2=X*Theta1.T
    A2=sigmoid(Z2)
    A2=np.insert(A2,0,1,axis=1)
    Z3=A2*Theta2.T
    A3=sigmoid(Z3)

    for i in range(m):
        yi=((y[i]==np.arange(num_labels))*1.0).reshape((num_labels,1))
        J=J-np.log(A3[i,:])*yi-np.log(1-A3[i,:])*(1-yi)

    T1sq=np.multiply(Theta1[:,1:],Theta1[:,1:])
    T2sq=np.multiply(Theta2[:,1:],Theta2[:,1:])

    J=J+lamd/2.0*(np.sum(T1sq)+np.sum(T2sq))
    J=J/m

    for i in range(m):
        yi=(y[i]==np.arange(num_labels))*1.0
        yi=np.asmatrix(yi)
        delta3=A3[i,:]-yi
        delta2=np.multiply(delta3*Theta2,sigmoidGradient(np.insert(Z2[i,:],0,1)))
        #  print(delta2,delta2[0,1:]);input()
        Theta1_grad=Theta1_grad+delta2[0,1:].T*X[i,:]
        Theta2_grad=Theta2_grad+delta3.T*A2[i,:]

    T1=Theta1
    T2=Theta2
    T1[:,0]=0
    T2[:,0]=0
    Theta1_grad=Theta1_grad/m+lamd*T1/m
    Theta2_grad=Theta2_grad/m+lamd*T2/m

    #  print(Theta1_grad.shape,Theta2_grad.shape);input()

    grad=np.r_[Theta1_grad.T.ravel().T,Theta2_grad.T.ravel().T]
    return J,grad

def nnJ(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamd):
    Theta1 = np.matrix(np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1),order='F'))
    Theta2 = np.matrix(np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1),order='F'))
    m=X.shape[0]
    J=0.0

    X=np.insert(X,0,1,axis=1)
    X=np.asmatrix(X)
    Z2=X*Theta1.T
    A2=sigmoid(Z2)
    A2=np.insert(A2,0,1,axis=1)
    Z3=A2*Theta2.T
    A3=sigmoid(Z3)

    for i in range(m):
        yi=((y[i]==np.arange(num_labels))*1.0).reshape((num_labels,1))
        J=J-np.log(A3[i,:])*yi-np.log(1-A3[i,:])*(1-yi)

    T1sq=np.multiply(Theta1[:,1:],Theta1[:,1:])
    T2sq=np.multiply(Theta2[:,1:],Theta2[:,1:])

    J=J+lamd/2.0*(np.sum(T1sq)+np.sum(T2sq))
    J=J/m

    J=np.asarray(J)
    return J.reshape((J.shape[0],))

def nnJprime(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamd):
    Theta1 = np.matrix(np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1),order='F'))
    Theta2 = np.matrix(np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1),order='F'))
    m=X.shape[0]
    Theta1_grad=np.zeros(Theta1.shape)
    Theta2_grad=np.zeros(Theta2.shape)

    X=np.insert(X,0,1,axis=1)
    X=np.asmatrix(X)
    Z2=X*Theta1.T
    A2=sigmoid(Z2)
    A2=np.insert(A2,0,1,axis=1)
    Z3=A2*Theta2.T
    A3=sigmoid(Z3)


    for i in range(m):
        yi=(y[i]==np.arange(num_labels))*1.0
        yi=np.asmatrix(yi)
        delta3=A3[i,:]-yi
        delta2=np.multiply(delta3*Theta2,sigmoidGradient(np.insert(Z2[i,:],0,1)))
        #  print(delta2,delta2[0,1:]);input()
        Theta1_grad=Theta1_grad+delta2[0,1:].T*X[i,:]
        Theta2_grad=Theta2_grad+delta3.T*A2[i,:]

    T1=Theta1
    T2=Theta2
    T1[:,0]=0
    T2[:,0]=0
    Theta1_grad=Theta1_grad/m+lamd*T1/m
    Theta2_grad=Theta2_grad/m+lamd*T2/m

    #  print(Theta1_grad.shape,Theta2_grad.shape);input()

    grad=np.r_[Theta1_grad.T.ravel().T,Theta2_grad.T.ravel().T]

    Jp=np.asarray(grad)
    return Jp.reshape((Jp.shape[0],))

def computeNumerialGradient(options,theta):
    eps=1e-4
    numgrad=np.zeros(theta.shape)
    perturb=np.zeros(theta.shape)
    for p in range(np.size(theta)):
        perturb[p]=eps
        loss1=nnCostFunction(theta-perturb,*options)[0]
        loss2=nnCostFunction(theta+perturb,*options)[0]
        numgrad[p]=(loss2-loss1)/(2*eps)
        perturb[p]=0
    return np.asmatrix(numgrad).T

def predictnn(Theta1,Theta2,X):
    m=X.shape[0]
    num_labels=Theta2.shape[0]
    p=np.zeros((m,1))
    X=np.asmatrix(X)
    h1=sigmoid(np.insert(X,0,1,axis=1)*Theta1.T)
    h2=sigmoid(np.insert(h1,0,1,axis=1)*Theta2.T)
    p=np.argmax(h2,axis=1)
    return p

