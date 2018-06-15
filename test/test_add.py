import numpy as np

from utils import gtensor, dm

def test_vector_vector():
  t1 = gtensor(dm('1 2'))
  t2 = gtensor(dm('3 4'))
  (t1 + t2).sum().backward()
  grad = dm('1 1')
  np.testing.assert_allclose(t1.grad, grad)
  np.testing.assert_allclose(t2.grad, grad)

def test_matrix_matrix():
  t1 = gtensor(dm('1 2 3; 3 4 5'))
  t2 = gtensor(dm('3 4 5; 5 6 7'))
  (t1 + t2).sum().backward()
  grad = dm('1 1 1; 1 1 1')
  np.testing.assert_allclose(t1.grad, grad)
  np.testing.assert_allclose(t2.grad, grad)

def test_matrix_vector():
  t1 = gtensor(dm('1 2 3; 3 4 5'))
  t2 = gtensor(dm('2 3 4'))
  (t1 + t2).sum().backward()
  grad1 = dm('1 1 1; 1 1 1')
  grad2 = dm('2 2 2')
  np.testing.assert_allclose(t1.grad, grad1)
  np.testing.assert_allclose(t2.grad, grad2)

def test_matrix_scalar():
  t1 = gtensor(dm('1 2 3; 3 4 5'))
  t2 = 2
  (t1 + t2).sum().backward()
  grad = dm('1 1 1; 1 1 1')
  np.testing.assert_allclose(t1.grad, grad)

def test_vector_scalar():
  t1 = gtensor(dm('1 2 3'))
  t2 = 2
  (t1 + t2).sum().backward()
  grad = dm('1 1 1')
  np.testing.assert_allclose(t1.grad, grad)