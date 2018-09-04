from theano import function
from theano import pp 
import theano.tensor as T
import numpy as np 

a,b = T.dmatrices('a','b')
diff = a-b
abs_diff = abs(diff)
diff_squared = diff**2
f = function([a,b], [diff, abs_diff, diff_squared])

mat1 = np.array([[1, 1], [1, 1]])
mat2 = np.array([[0, 1], [2, 3]])

out = f(mat1, mat2)
print out