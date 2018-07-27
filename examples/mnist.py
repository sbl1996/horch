import argparse

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

import horch as H
from horch.utils import standardize, split, evaluate
from horch.optim import SGD

from models import MLP

parser = argparse.ArgumentParser(description='Horch MNIST Training')
parser.add_argument('--batch-norm', '-bn', action='store_true', help='enable batch normalization')
args = parser.parse_args()

DATA_HOME = 'D:\\MLCode\\datasets'
mnist = fetch_mldata('MNIST original', data_home=DATA_HOME)
X = mnist.data
y = mnist.target.astype(np.int)

indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

m_train = 59000
x_train = X[:m_train]
y_train = y[:m_train]
x_val = X[m_train:60000]
y_val = y[m_train:60000]
x_test = X[60000:]
y_test = y[60000:]

net = MLP([784, 'bn', 'relu', 100, 'bn', 'relu', 10])
optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)

batch_size = 32
epochs = 5
for epoch in range(epochs):
  print("Epoch %d" % (epoch + 1))
  batches = split(x_train, y_train, batch_size)
  for batch in batches:
    input, target = batch
    input = H.tensor(input)
    target = H.tensor(target)

    net.zero_grad()
    output = net(input)
    loss = H.CrossEntropyLoss(output, target)
    loss.backward()
    optimizer.step()
  val_acc, val_loss = evaluate(net, x_val, y_val)
  print("val_loss: %.4f   val_acc: %.2f" % (val_loss, val_acc))
train_acc, train_loss = evaluate(net, x_train, y_train)
test_acc, test_loss = evaluate(net, x_test, y_test)
print("%d Epochs." % epochs)
print("Training set:")
print("Loss: %.4f   Accuracy: %.2f" % (train_loss, train_acc))
print("Test set:")
print("Loss: %.4f   Accuracy: %.2f" % (test_loss, test_acc))