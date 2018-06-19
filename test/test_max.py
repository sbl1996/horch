import horch as H

import numpy as np

from utils import gtensor, dm

def test_axis_none():
  t1 = gtensor(dm('1 2 3; 4 5 6'))
  k = 3
  (H.max(t1) * k).backward()
  grad = dm('0 0 0; 0 0 3')
  np.testing.assert_allclose(t1.grad, grad)

def test_axis():
  t1 = gtensor(dm('1 2 3; 4 5 6'))
  k = gtensor(dm('3 6'))
  (H.max(t1, axis=1) * k).sum().backward()
  grad = dm('0 0 3; 0 0 6')
  np.testing.assert_allclose(t1.grad, grad)

def test_high_dim():
  d = [[[ 0,  8,  0,  5],
        [ 7,  1,  7,  1],
        [ 9,  1,  6,  9]],
       [[ 7,  0,  6,  2],
        [ 8,  5,  2,  5],
        [ 8,  1,  4,  6]]]
  a = gtensor(dm(d))
  a.max(axis=1).sum().backward()
  print(a.grad)
  g = [[
    [ 0.,  1.,  0.,  0.],
    [ 0.,  0.,  1.,  0.],
    [ 1.,  0.,  0.,  1.]],

   [[ 0.,  0.,  1.,  0.],
    [ 1.,  1.,  0.,  0.],
    [ 0.,  0.,  0.,  1.]]
  ]
  grad = dm(g)
  np.testing.assert_allclose(a.grad, grad)