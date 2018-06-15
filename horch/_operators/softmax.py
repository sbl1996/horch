# import numpy as np

# from .operator import Operator

# class Softmax(Operator):

#   def __init__(self, parents, *args):
#     super(Softmax, self).__init__(parents, args)

#   def forward(self, x, axis=None):
#     m = x.max(axis=axis, keepdims=True)
#     ex = np.exp(x - m)
#     return ex / ex.sum(axis=axis, keepdims=True)

#   def backward(self, acc, x, axis):
    
#     return (1 - d) * d