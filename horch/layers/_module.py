from collections import OrderedDict

from horch import Tensor
from horch import Parameter

class Module:

  def __init__(self):
    self._parameters = OrderedDict()
    self._children = OrderedDict()

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def named_parameters(self, mem=None, prefix=''):
    if mem is None:
      mem = set()
    for name, p in self._parameters.items():
      if p not in mem:
        mem.add(p)
        yield prefix + ('.' if prefix else '') + name, p
    for mname, module in self.named_children():
      sub_prefix = prefix + ('.' if prefix else '') + mname
      for name, p in module.named_parameters(mem, sub_prefix):
        yield name, p

  def parameters(self):
    for name, param in self.named_parameters():
      yield param

  def named_children(self):
    for name, module in self._children.items():
      yield name, module

  def children(self):
    for name, child in self.named_children():
      yield child

  def __setattr__(self, name, value):
    params = self.__dict__.get('_parameters')
    if isinstance(value, Parameter):
      params[name] = value
    elif isinstance(value, Module):
      self._children[name] = value
    else:
      object.__setattr__(self, name, value)

  def __getattr__(self, name):
    if '_parameters' in self.__dict__:
        params = self.__dict__['_parameters']
        if name in params:
            return params[name]
    if '_children' in self.__dict__:
        children = self.__dict__['_children']
        if name in children:
            return children[name]
    raise AttributeError("'{}' object has no attribute '{}'".format(
        type(self).__name__, name))

  def __dir__(self):
    module_attrs = dir(self.__class__)
    attrs = list(self.__dict__.keys())
    parameters = list(self._parameters.keys())
    children = list(self._children.keys())
    keys = module_attrs + attrs + parameters + children

    keys = [key for key in keys if not key[0].isdigit()]

    return sorted(keys)

  def zero_grad(self):
    for p in self.parameters():
      p.zero_grad()
