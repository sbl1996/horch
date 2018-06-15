import argparse

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

import horch as H
from horch.utils import standardize, split, evaluate
from horch.optim import SGD

from hidden_net import HiddenNet

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

X = X / 255

m_train = 60000
m_test = 10000
x_train = X[:m_train]
y_train = y[:m_train]
x_val = X[m_train:]
y_val = y[m_train:]
x_test = X[60000: 60000 + m_test]
y_test = y[60000: 60000 + m_test]

net = HiddenNet(784, 100, 10, args.batch_norm)
optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)

batch_size = 32
epochs = 5
for epoch in range(epochs):
  print("Epoch %d" % epoch)
  batches = split(x_train, y_train, batch_size)
  for batch in batches:
    inputs, target = batch
    inputs = H.tensor(inputs)
    target = H.tensor(target)

    net.zero_grad()
    output = net(inputs)
    loss = H.CrossEntropyLoss(output, target)
    loss.backward()
    optimizer.step()
  val_loss, val_acc = evaluate(net, x_val, y_val)
  print("val_loss: %.4f   val_acc: %.2f" % (val_loss, val_acc))
train_loss, train_acc = evaluate(net, x_train, y_train)
test_loss, test_acc = evaluate(net, x_test, y_test)
print("%d Epochs." % epochs)
print("Training set:")
print("Loss: %.4f   Accuracy: %.2f" % (train_loss, train_acc))
print("Test set:")
print("Loss: %.4f   Accuracy: %.2f" % (test_loss, test_acc))