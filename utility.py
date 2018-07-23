import torch
import numpy as np
from torch.autograd import Variable


def shuffle(x, y):
        index = np.random.permutation(x.shape[0])
        return x[index, :, :], y[index, :, :]


def train(x, y, batch_size, optimizer, criterion, model, cuda):

    model.train()
    batch_cost = []
    x_b, y_b = shuffle(x, y)
    n_train_batch = x.shape[0] // batch_size

    for i in range(n_train_batch):

        if cuda:
            x_tr = Variable(torch.from_numpy(x_b)).cuda()
            y_tr = Variable(torch.from_numpy(y_b)).cuda()
        else:
            x_tr = Variable(torch.from_numpy(x_b))
            y_tr = Variable(torch.from_numpy(y_b))

        output, state = model(x_tr[i*batch_size:(i+1)*batch_size])
        loss = criterion(output, y_tr[i*batch_size:(i+1)*batch_size])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_cost.append(loss.cpu().data.numpy())

    return np.mean(batch_cost)


def validate(x, y, batch_size, criterion, model, cuda):

    model.eval()
    batch_cost = []
    x_b, y_b = shuffle(x, y)
    n_valid_batch = x.shape[0] // batch_size

    for i in range(n_valid_batch):

        if cuda:
            x_va = Variable(torch.from_numpy(x_b)).cuda()
            y_va = Variable(torch.from_numpy(y_b)).cuda()
        else:
            x_va = Variable(torch.from_numpy(x_b))
            y_va = Variable(torch.from_numpy(y_b))

        output, state = model(x_va[i*batch_size:(i+1)*batch_size])
        loss = criterion(output, y_va[i*batch_size:(i+1)*batch_size])
        batch_cost.append(loss.cpu().data.numpy())

    return np.mean(batch_cost)


def test(x, y, batch_size, criterion, model, cuda):

    model.eval()
    batch_cost = []
    x_b, y_b = shuffle(x, y)
    n_valid_batch = x.shape[0] // batch_size

    for i in range(n_valid_batch):

        if cuda:
            x_va = Variable(torch.from_numpy(x_b)).cuda()
            y_va = Variable(torch.from_numpy(y_b)).cuda()
        else:
            x_va = Variable(torch.from_numpy(x_b))
            y_va = Variable(torch.from_numpy(y_b))

        output, state = model(x_va[i*batch_size:(i+1)*batch_size])
        loss = criterion(output, y_va[i*batch_size:(i+1)*batch_size])
        batch_cost.append(loss.cpu().data.numpy())

    return np.mean(batch_cost)


class Early_Stopper():
    def __init__(self, patience):

        self.n_epochs = 0
        self.final_epoch = 0
        self.patience = patience
        self.keep_training = True
        self.best_validation = 10000

    def update(self, validation, epoch):

        if self.best_validation > validation:
            self.best_validation = validation
            self.n_epochs = 0
        if self.best_validation < validation and self.patience >= self.n_epochs:
            self.n_epochs += 1
        if self.best_validation < validation and self.patience < self.n_epochs:
            self.keep_training = False
            self.final_epoch = epoch
