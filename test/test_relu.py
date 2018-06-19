import numpy as np

import horch as H

from utils import gtensor, dm

def test_sigmoid():
  a = gtensor(dm('1 -2; -3 4'))
  k = gtensor(dm('8  3;  2 1'))
  
  b = H.relu(a)
  (b * k).sum().backward()

  grad = dm('8 0; 0 1')
  np.testing.assert_allclose(a.grad, grad)