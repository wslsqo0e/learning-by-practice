import sys
sys.path.insert(0, "../")
from d2l import data_iter
from matplotlib import pyplot as plt
import torch
import random
import numpy as np
import os

def seed_torch(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch()

num_inputs = 2
num_examples = 1000
true_w = torch.tensor([2, -3.4])
#true_w = torch.tensor([[2], [-3.4]])
true_b = 4.2
# normal_ 表示正太分布初始化
features = torch.zeros(size=(num_examples, num_inputs)).normal_()
labels = torch.matmul(features, true_w) + true_b
labels += torch.zeros(size=labels.shape).normal_(std=0.01)

batch_size = 10

w = torch.zeros(size=(num_inputs, 1)).normal_(std=0.01)
#w = torch.zeros(size=(num_inputs, 1))
b = torch.zeros(size=(1,))

w.requires_grad_(True)
b.requires_grad_(True)

# This function has been saved in the d2l package for future use
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# This function has been saved in the d2l package for future use
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# This function has been saved in the d2l package for future use
def sgd(params, lr, batch_size):
    for param in params:
        #param.data.sub_(lr*param.grad/batch_size)
        param.data.sub_(lr*param.grad)
        param.grad.data.zero_()

lr = 0.03  # Learning rate
num_epochs = 3  # Number of iterations
net = linreg  # Our fancy linear model
loss = squared_loss  # 0.5 (y-y')^2

for epoch in range(num_epochs):
    # Assuming the number of examples can be divided by the batch size, all
    # the examples in the training data set are used once in one epoch
    # iteration. The features and tags of mini-batch examples are given by X
    # and y respectively
    count = 0
    for X, y in data_iter(batch_size, features, labels):
        count += 1
        l = loss(net(X, w, b), y)  # Minibatch loss in X and y
        #l.mean().backward()  # Compute gradient on l with respect to [w,b]
        l.sum().backward()  # Compute gradient on l with respect to [w,b]
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().numpy()))

print('Error in estimating w', true_w - w.reshape(true_w.shape))
print('Error in estimating b', true_b - b)
