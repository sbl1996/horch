# Horch

Horch is a Python toy project inspired by PyTorch that provides one main feature:
- Automatic Differentiation for NumPy

## Requirements
- Python 3 (only 3.6 is tested)
- numpy
- pytest (for test)
- sklearn (to run examples)

## Examples
You can see some examples in examples.

- iris
- mnist


## Operators
Operators below are implemented, but only some of them are tested.

- abs 
- add
- exp
- getitem
- log (*Untested*)
- matmul
- max (*Untested*)
- maximum
- mean (*Untested*)
- mul
- neg
- relu (*Untested*)
- reshape (*Untested*)
- sigmoid (*Untested*)
- softmax (*Untested*)
- sub 
- sum (*Untested*)

## Modules
Horch consists of the following components:

- horch: tensor and operators on it
- optim: optimizers to train network (now only SGD)
- layers: common deep neural network components
- utils: miscellaneous functions for data preprocessing, model evaluating and others

## Imports
Import statements below are recommended.
```python
import horch as H
import horch.layers as L
```