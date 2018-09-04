from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from theano import pp 
import numpy as np
import theano.tensor as T

srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))   # 2x2 matrix of random draws
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True) 
nearly_zeros = function([], rv_u + rv_u - 2*rv_u) 

f_val0 = f()
f_val1 = f()   # different random uniform values

g_val0 = g()
g_val1 = g()  # same values as g_val0

state_after_v0 = rv_u.rng.get_value().get_state()
nearly_zeros()
v1 = f()
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow=True)
v2 = f()             # v2 != v1
v3 = f()             # v3 == v1
