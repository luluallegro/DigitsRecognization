'''
#  ======================================================================
#     FileName: HandWritten.py
#      Project: HandDigitRecognization
#       Author: Yong Yang, Department of Mathematics, UTA
#        Email: yongyang@mavs.uta.edu
#      Created: 2017-02-20 10:57:10
#   LastChange: 2017-02-20 21:42:20
#  ======================================================================
'''
import scipy.io as sio
import random as rd
from subroutines import *

input_lay_size=400  # 20x20 Input Images of Digits
num_labels=10       # 10 labels, from 0 to 9

# Loading and Visualizing Traning Data
print('Loading and Visualizing Data...\n')
mat_contents=sio.loadmat('ex3data1.mat')
X=mat_contents['X']
y=mat_contents['y']
y[y==10]=0
m=X.shape[0]

#Randomly selet 100 data points to display
rand_indices=rd.sample(range(m),100)
sel=X[rand_indices,:]
displayData(sel)

pause=input('Program paused. Press enter to continue.')

print('Training One-vs-All Logistic Regression...\n')
lamd=0.1
all_theta=oneVsAll(X,y,num_labels,lamd)
pred=predictOneVsAll(all_theta,X).reshape(y.shape)
accuracy=np.mean((pred==y)*1.0)*100
print('Traning Set Accuracy:',accuracy,'%')
