import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

import horch as H
from horch.utils import standardize, split, evaluate
from horch.optim import SGD

from hidden_net import HiddenNet

DATA_HOME = 'D:\\MLCode\\datasets'
mnist = fetch_mldata('MNIST original', data_home=DATA_HOME)
X = mnist.data / 255
y = mnist.target.astype(np.int)

m_train = 10000
m_test = 1000
train_indices = np.random.choice(60000, size=m_train, replace=False)
test_indices = np.random.choice(10000, size=m_test, replace=False) + 60000
x_train = X[train_indices]
y_train = y[train_indices]
x_test = X[test_indices]
y_test = y[test_indices]

net = HiddenNet(784, 100, 10)
optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)

epochs = 30
for epoch in range(epochs):
  batches = split(x_train, y_train, 32)
  for batch in batches:
    inputs, target = batch
    inputs = H.tensor(inputs.reshape(-1, 784))
    target = H.tensor(target)

    net.zero_grad()
    output = net(inputs)
    loss = H.CrossEntropyLoss(output, target)
    loss.backward()
    optimizer.step()
train_loss, train_acc = evaluate(net, x_train, y_train)
test_loss, test_acc = evaluate(net, x_test, y_test)
print("Training set:")
print("Loss: %.4f   Accuracy: %.2f" % (train_loss, train_acc))
print("Test set:")
print("Loss: %.4f   Accuracy: %.2f" % (test_loss, test_acc))