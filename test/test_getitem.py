import numpy as np

from utils import gtensor, dm

def test_vector():
  t1 = gtensor(dm('1 2 3; 4 5 6'))
  t2 = t1[1]
  t2.sum().backward()
  grad = dm('0 0 0; 1 1 1')
  np.testing.assert_allclose(t1.grad, grad)

def test_vector():
  t1 = gtensor(dm('1 2 3; 4 5 6'))
  t2 = t1[1, 1]
  (t2 + 1).sum().backward()
  grad = dm('0 0 0; 0 1 0')
  np.testing.assert_allclose(t1.grad, grad)