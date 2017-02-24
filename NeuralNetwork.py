'''
#  ======================================================================
#     FileName: NeuralNetwork.py
#      Project: DigitsRecognization
#       Author: Yong Yang, Department of Mathematics, UTA
#        Email: yongyang@uta.edu; yongyang.math@gmail.com
#      Created: 2017-02-22 16:24:32
#   LastChange: 2017-02-24 12:36:44
#  ======================================================================
'''
import scipy.io as sio
import random as rd
from scipy import optimize
from subroutines import *

input_layer_size=400  # 20x20 Input Images of Digits
hidden_layer_size=25 # 25 hidden units
num_labels=10       # 10 labels, from 0 to 9

# Loading and Visualizing Traning Data
print('Loading and Visualizing Data...\n')
mat_contents=sio.loadmat('ex3data1.mat')
X=mat_contents['X']
y=mat_contents['y']
y[y==10]=0
m=X.shape[0]

# Randomly selet 100 data points to display
rand_indices=rd.sample(range(m),100)
sel=X[rand_indices,:]
#  displayData(sel)

print('Initializing Neural Network Parameters...')
initial_Theta1=randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2=randInitializeWeights(hidden_layer_size,num_labels)

# Unroll Parameters
initial_nn_params=np.r_[initial_Theta1.T.ravel().T,initial_Theta2.T.ravel().T]

# Check gradients by running checkNNGradients
#  print('Checking Backpropagation...')
#  diff=checkNNGradients(3)
#  print('If your backpropagation implementation is correct, then \n',
         #  'the relative difference will be small (less than 1e-9). \n',
         #  '\nRelative Difference:',diff )

print('\nTraining Neural Network...\n')
lamd=1
options=(input_layer_size,hidden_layer_size,num_labels,X,y,lamd)
nn_params=optimize.fmin_cg(nnJ,initial_nn_params,nnJprime,options,disp=True,gtol=1e-5,maxiter=50)

Theta1 = np.matrix(np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1),order='F'))
Theta2 = np.matrix(np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1),order='F'))

print('\nVisualizing Neural Network...\n')
#  displayData(Theta1[:,1:])
pred=predictnn(Theta1,Theta2,X)

accuracy=np.mean((pred==y)*1.0)*100
print('Traning Set Accuracy:',accuracy,'%')

rp=rd.sample(range(m),m)

X=np.asmatrix(X)
for i in range(m):
    print('\nDisplaying Example Image\n')
    displayData(X[rp[i],:])

    print('\nNeural Network predition:',pred[rp[i]])
    flag=input('Exit? yes/no\n')
    if flag=='yes': break

