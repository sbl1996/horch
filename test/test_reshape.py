import numpy as np

from utils import gtensor, dm

def test_2d_1d():
  t1 = gtensor(dm('1 2; 4 5'))
  shape = (4,)
  t2 = t1.reshape(shape)
  t2.sum().backward()
  grad = dm('1 1; 1 1')
  np.testing.assert_allclose(t1.grad, grad)

def test_1d_2d():
  t1 = gtensor(dm('1 2 4 5'))
  t2 = t1.reshape(2, 2)
  t2.sum().backward()
  grad = dm('1 1 1 1')
  np.testing.assert_allclose(t1.grad, grad)

def test_nd_nd():
  shape = (1,2,3,4)
  t1 = gtensor(np.random.rand(*shape))
  t2 = t1.reshape(8, 3)
  t2.sum().backward()
  grad = np.ones(shape)
  np.testing.assert_allclose(t1.grad, grad)