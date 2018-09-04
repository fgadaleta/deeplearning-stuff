from theano import function
from theano import pp 
import theano.tensor as T
import numpy as np 

x = T.dmatrix('x')
s = 1/(1+T.exp(-x))
logistic = function([x],s)
mat = np.array([[0,1],[-1,-2]])
out = logistic(mat)
print out


