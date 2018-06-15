import numpy as np

from utils import gtensor, dm

def test_neg():
  t1 = gtensor(dm('1 2'))
  t2 = -t1
  t2.sum().backward()
  grad = dm('-1 -1')
  np.testing.assert_allclose(t1.grad, grad)