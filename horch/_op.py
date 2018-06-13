from . import _operators
from .utils import check_tensor as _check_tensor

def abs(t):
  t = _check_tensor(t)
  op = _operators.Abs([t.op])
  return op.tensor

def add(lt, rt):
  lt = _check_tensor(lt)
  rt = _check_tensor(rt)
  op = _operators.Add([lt.op, rt.op])
  return op.tensor

def exp(t):
  t = _check_tensor(t)
  op = _operators.Exp([t.op])
  return op.tensor

def getitem(t, ind):
  t = _check_tensor(t)
  op = _operators.GetItem([t.op], ind)
  return op.tensor

def log(t):
  t = _check_tensor(t)
  op = _operators.Log([t.op])
  return op.tensor

def mean(t, axis=None):
  t = _check_tensor(t)
  op = _operators.Mean([t.op], axis)
  return op.tensor

def mul(lt, rt):
  lt = _check_tensor(lt)
  rt = _check_tensor(rt)
  op = _operators.Mul([lt.op, rt.op])
  return op.tensor

def maximum(lt, rt):
  lt = _check_tensor(lt)
  rt = _check_tensor(rt)
  op = _operators.Maximum([lt.op, rt.op])
  return op.tensor

def matmul(lt, rt):
  lt = _check_tensor(lt)
  rt = _check_tensor(rt)
  op = _operators.MatMul([lt.op, rt.op])
  return op.tensor

def max(t, axis=None, keepdims=False):
  t = _check_tensor(t)
  op = _operators.Max([t.op], axis, keepdims)
  return op.tensor

def neg(t):
  t = _check_tensor(t)
  op = _operators.Neg([t.op])
  return op.tensor

def relu(t):
  t = _check_tensor(t)
  op = _operators.ReLU([t.op])
  return op.tensor

def reshape(t, *shapes):
  t = _check_tensor(t)
  op = _operators.Reshape([t.op], shapes)
  return op.tensor  

def sum(t, axis=0):
  t = _check_tensor(t)
  op = _operators.Sum([t.op], axis)
  return op.tensor

def sub(lt, rt):
  lt = _check_tensor(lt)
  rt = _check_tensor(rt)
  op = _operators.Sub([lt.op, rt.op])
  return op.tensor

def sigmoid(t):
  t = _check_tensor(t)
  op = _operators.Sigmoid([t.op])
  return op.tensor