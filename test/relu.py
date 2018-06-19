import numpy as np

from utils import gtensor, dm

def test_relu():
  t1 = gtensor(dm('-1 0 2'))
  t2 = t1.relu()
  t2.sum().backward()
  grad = dm('0 0 1')
  np.testing.assert_allclose(t1.grad, grad)