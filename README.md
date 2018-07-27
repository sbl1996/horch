# Horch

Horch is a Python toy project inspired by PyTorch and Chainer that provides one main feature:
- Automatic Differentiation for NumPy

## Requirements
- Python 3 (only 3.6 is tested)
- NumPy

## Examples
You can see some examples in [examples](https://github.com/sbl1996/horch/tree/master/examples).

### Requirements
- scikit-learn
- torch

### Models
- MLP
- LeNet
- LeNetPlus (with batch normalization and dropout)

### Datasets
- [breast_cancer](https://github.com/sbl1996/horch/blob/master/examples/breast_cancer.py)
- [iris](https://github.com/sbl1996/horch/blob/master/examples/iris.py)
- [mnist](https://github.com/sbl1996/horch/blob/master/examples/mnist.py)
- [fmnist](https://github.com/sbl1996/horch/blob/master/examples/mnist.py)
- [cifar10](https://github.com/sbl1996/horch/blob/master/examples/cifar10.py)


## Operators

### Implemented
- abs 
- add
- conv2d
- div
- exp
- getitem
- log
- matmul
- max
- maximum
- max_pool2d
- mean
- mul
- neg
- relu
- reshape
- sigmoid
- sqrt
- std
- sub 
- sum
- var

### Planned
- Group Convolution
- Average Pool
- Variants of ReLU
- Tanh

## Modules
Horch consists of the following components:

- horch: tensor and operators
- optim: optimizers (now only SGD)
- layers: common deep neural network components
- utils: misc for data preparing and model evaluating

## Imports
Import statements below are recommended.
```python
import horch as H
import horch.layers as L
```