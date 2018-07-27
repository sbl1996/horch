import horch as H

import numpy as np

from utils import gtensor, dm

def test_axis_none():
  t1 = gtensor(dm('1 2 3; 4 5 6'))
  k = 3
  (H.mean(t1, axis=None) * k).backward()
  grad = dm('.5 .5 .5; .5 .5 .5')
  np.testing.assert_allclose(t1.grad, grad)

def test_axis():
  t1 = gtensor(dm('1 2 3; 4 5 6'))
  k = gtensor(dm('3 6'))
  (H.mean(t1, axis=1) * k).sum().backward()
  grad = dm('1 1 1; 2 2 2')
  np.testing.assert_allclose(t1.grad, grad)

def test_multi_axes():
  a = [
    [[  4., -10.,   7.,  24.],
     [ -1.,   4.,  -6.,  11.],
     [  8.,  11.,  -8.,   8.]],

    [[  3., -10.,   2.,  -5.],
     [ 10.,  18.,  -5., -15.],
     [ -1.,  10., -13.,  17.]]
  ]
  a = np.array(a)
  ha = gtensor(a)
  # k = [
  #   [[ 8.,  6.,  8., 10.],
  #    [ 9.,  7.,  1.,  8.],
  #    [ 3.,  2.,  6.,  6.]],

  #   [[ 7.,  7.,  1.,  5.],
  #    [ 3.,  5.,  5.,  3.],
  #    [ 7.,  9.,  5., 10.]]
  # ]
  k = [ 1., 2., 4. ]
  k = np.array(k)
  hk = gtensor(k)
  (H.mean(ha, axis=(0,2)) * hk).sum().backward()
  grad = [
    [[ 0.125, 0.125, 0.125, 0.125],
     [ 0.25 , 0.25 , 0.25 , 0.25 ],
     [ 0.5  , 0.5  , 0.5  , 0.5  ]],
    
    [[ 0.125, 0.125, 0.125, 0.125],
     [ 0.25 , 0.25 , 0.25 , 0.25 ],
     [ 0.5  , 0.5  , 0.5  , 0.5  ]]
  ]
  grad = np.array(grad)
  np.testing.assert_allclose(ha.grad, grad)