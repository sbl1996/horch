import numpy as np

import horch as H

from utils import gtensor, dm

def test_elu():
  a = gtensor(dm('1 -2; -3 4'))
  k = gtensor(dm('8  3;  2 1'))
  
  b = H.elu(a)
  ret = (b * k + b).sum()
  ret.backward()

  grad = dm([
    [9.        , 0.54134113],
    [0.14936121, 2.        ],
  ])
  np.testing.assert_allclose(a.grad, grad)