from theano import function
from theano import pp 
import theano.tensor as T
import numpy as np 

a = T.vector() # declare variable
b = T.vector()
out = a ** 2 + b ** 2 + 2 * a * b  # build symbolic expression
f = function([a,b], out)  		   # compile function

print f([1, 1], [4,5])   # vectors as input

