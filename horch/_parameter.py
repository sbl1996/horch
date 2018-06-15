from horch.tensor import Tensor

class Parameter(Tensor):

  def __init__(self, data):
    super(Parameter, self).__init__(data, requires_grad=True)