import numpy as np

import horch as H

from utils import gtensor, dm

def test_relu():
  a = gtensor(dm('1 -2; -3 4'))
  k = gtensor(dm('8  3;  2 1'))
  
  b = H.relu(a)
  ret = (b * k + b).sum()
  ret.backward()

  grad = dm('9 0; 0 2')
  np.testing.assert_allclose(a.grad, grad)