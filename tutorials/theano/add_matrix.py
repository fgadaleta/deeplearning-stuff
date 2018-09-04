from theano import function
from theano import pp 
import theano.tensor as T
import numpy as np 

print "adding two scalars"
x = T.dscalar('x')   # name not mandatory but good for debug
y = T.dscalar('y')
z = x+y
pp(z)  #pretty print expression
f = function([x,y], z)  # takes a while (compile to C)
out = f(12.2, 14.5)
print out


print "adding two matrices"
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x+y
f = function([x,y],z)

mat1 = np.array([[1,2,5],[3,4,7]])
mat2 = np.array([[10,3,2],[4,2,5]])
out = f(mat1,mat2)
print out





