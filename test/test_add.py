import numpy as np

from utils import gtensor, dm

def test_add():
  # vector add vector
  t1 = gtensor(dm('1 2'))
  t2 = gtensor(dm('3 4'))
  (t1 + t2).sum().backward()
  grad = dm('1 1')
  assert np.array_equal(t1.grad, grad)
  assert np.array_equal(t2.grad, grad)

  # matrix add matrix
  t1 = gtensor(dm('1 2 3; 3 4 5'))
  t2 = gtensor(dm('3 4 5; 5 6 7'))
  (t1 + t2).sum().backward()
  grad = dm('1 1 1; 1 1 1')
  assert np.array_equal(t1.grad, grad)
  assert np.array_equal(t2.grad, grad)

  # matrix add vector
  t1 = gtensor(dm('1 2 3; 3 4 5'))
  t2 = gtensor(dm('2 3 4'))
  (t1 + t2).sum().backward()
  grad1 = dm('1 1 1; 1 1 1')
  grad2 = dm('2 2 2')
  assert np.array_equal(t1.grad, grad1)
  assert np.array_equal(t2.grad, grad2)

  # matrix add scalar
  t1 = gtensor(dm('1 2 3; 3 4 5'))
  t2 = 2
  (t1 + t2).sum().backward()
  grad = dm('1 1 1; 1 1 1')
  assert np.array_equal(t1.grad, grad)

  # vector add scalar
  t1 = gtensor(dm('1 2 3'))
  t2 = 2
  (t1 + t2).sum().backward()
  grad = dm('1 1 1')
  assert np.array_equal(t1.grad, grad)