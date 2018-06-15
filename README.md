# Horch

Horch is a Python toy project inspired by PyTorch that provides one main feature:
- Automatic Differentiation for NumPy

## Requirements
- Python 3 (only 3.6 is tested)
- NumPy
- pytest (for test)
- scikit-learn (to run examples)

## Examples
You can see some examples in [examples](https://github.com/sbl1996/horch/tree/master/examples).

- [iris](https://github.com/sbl1996/horch/blob/master/examples/iris.py)
- [mnist](https://github.com/sbl1996/horch/blob/master/examples/mnist.py)
- [breast_cancer](https://github.com/sbl1996/horch/blob/master/examples/breast_cancer.py)
- [cifar10](https://github.com/sbl1996/horch/blob/master/examples/cifar10.py)


## Operators
Operators below are implemented, but only some of them are tested.

- abs 
- add
- div
- exp
- getitem
- log (*Untested*)
- matmul
- max (*Untested*)
- maximum
- mean
- mul
- neg
- relu (*Untested*)
- reshape (*Untested*)
- sigmoid (*Untested*)
- sqrt
- std
- sub 
- sum
- var

## Modules
Horch consists of the following components:

- horch: tensor and operators on it
- optim: optimizers to train network (now only SGD)
- layers: common deep neural network components
- utils: misc for data preparing and model evaluating

## Imports
Import statements below are recommended.
```python
import horch as H
import horch.layers as L
```