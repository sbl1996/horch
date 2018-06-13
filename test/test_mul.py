import numpy as np

from utils import gtensor, dm

def test_mul():
  # vector mul vector
  t1 = gtensor(dm('1 2'))
  t2 = gtensor(dm('3 4'))
  (t1 * t2).sum().backward()
  assert np.array_equal(t1.grad, t2.data)
  assert np.array_equal(t2.grad, t1.data)

  # matrix mul matrix
  t1 = gtensor(dm('1 2 3; 3 4 5'))
  t2 = gtensor(dm('3 4 5; 5 6 7'))
  (t1 * t2).sum().backward()
  assert np.array_equal(t1.grad, t2.data)
  assert np.array_equal(t2.grad, t1.data)

  # matrix mul vector
  t1 = gtensor(dm('1 2 3; 3 4 5'))
  t2 = gtensor(dm('2 3 4'))
  (t1 * t2).sum().backward()
  grad1 = dm('2 3 4; 2 3 4')
  grad2 = dm('4 6 8')
  assert np.array_equal(t1.grad, grad1)
  assert np.array_equal(t2.grad, grad2)

  # vector mul matrix
  t1 = gtensor(dm('2 3 4'))
  t2 = gtensor(dm('1 2 3; 3 4 5'))
  (t1 * t2).sum().backward()
  grad1 = dm('4 6 8')
  grad2 = dm('2 3 4; 2 3 4')
  assert np.array_equal(t1.grad, grad1)
  assert np.array_equal(t2.grad, grad2)

  # matrix mul scalar
  t1 = gtensor(dm('1 2 3; 3 4 5'))
  t2 = 2
  (t1 * t2).sum().backward()
  grad = dm('2 2 2; 2 2 2')
  assert np.array_equal(t1.grad, grad)

  # scalar mul matrix
  t1 = 2
  t2 = gtensor(dm('1 2 3; 3 4 5'))
  (t1 * t2).sum().backward()
  grad = dm('2 2 2; 2 2 2')
  assert np.array_equal(t2.grad, grad)

  # vector mul scalar
  t1 = gtensor(dm('1 2 3'))
  t2 = 2
  (t1 * t2).sum().backward()
  grad = dm('2 2 2')
  assert np.array_equal(t1.grad, grad)

  # scalar mul vector
  t1 = 2
  t2 = gtensor(dm('1 2 3'))
  (t1 * t2).sum().backward()
  grad = dm('2 2 2')
  assert np.array_equal(t2.grad, grad)