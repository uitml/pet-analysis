# %%
import torch
import pandas
import datetime
import numpy as np
import torch.nn as nn
import scipy.io as sio
from LSTM_rnn import LSTM
from utility import train
import matplotlib.pyplot as plt
from torch.autograd import grad
from torch.autograd import Variable
from PET_graph_loader import proper_split
from PET_graph_loader import leave_one_out



path = '/home/kristoffer/scripts/data/PET_voistat_data_12_VOIs_n68_v7.mat'
#x_tr, y_tr, x_te, y_te = leave_one_out(path)
x_tr, y_tr, x_va, y_va, x_te, y_te = proper_split(path)
data = sio.loadmat(path)
names = []
for i in range(11):
    names.append(data['VOInames'][i+1][0][0])


# %%


n_epochs = 500
n_inputs = 11
n_layers = 2
n_hidden = 20
n_outputs = 1
batch_size = 11
bidirectional = False
time_start = datetime.datetime.now()

lstm = LSTM(n_inputs, n_hidden, n_outputs, n_layers, batch_size, bidirectional)

all_losses = []
test_loss = []
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters())

print('Training started at:', time_start)

for epoch in range(1, n_epochs + 1):

    all_losses.append(train(x_tr,
                            y_tr,
                            batch_size,
                            optimizer,
                            criterion,
                            lstm,
                            False))

    print(pandas.DataFrame([epoch,
                            all_losses[-1],
                            datetime.datetime.now()-time_start],
                           ['Iteration', 'Cost', 'Elapsed time'],
                           ['LSTM']))

    pred = lstm.pred(Variable(torch.from_numpy(x_te)))
    test_loss.append(np.abs((pred.data.numpy()-y_te)).sum() / batch_size)
    print(test_loss[-1])

plt.figure(8)
plt.plot(all_losses)
plt.plot(test_loss)

    pred = rnn.pred(Variable(torch.from_numpy(x_te)))
    test_loss.append(np.abs((pred.data.numpy()-y_te)).sum() / batch_size)
    print(test_loss[-1])

plt.figure(8)
plt.plot(all_losses)
plt.plot(test_loss)

# %%

pred = lstm.pred(Variable(torch.from_numpy(x_te)))

plt.figure(1, figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.plot(pred.data.numpy()[0, :, :], label='pred')
plt.plot(y_te[0, :, :], label='true')
plt.legend()
plt.subplot(2, 3, 2)
plt.plot(pred.data.numpy()[1, :, :], label='pred')
plt.plot(y_te[1, :, :], label='true')
plt.legend()
plt.subplot(2, 3, 3)
plt.plot(pred.data.numpy()[2, :, :], label='pred')
plt.plot(y_te[2, :, :], label='true')
plt.legend()
plt.subplot(2, 3, 4)
plt.plot(pred.data.numpy()[3, :, :], label='pred')
plt.plot(y_te[3, :, :], label='true')
plt.legend()
plt.subplot(2, 3, 5)
plt.plot(pred.data.numpy()[4, :, :], label='pred')
plt.plot(y_te[4, :, :], label='true')
plt.legend()
plt.subplot(2, 3, 6)
plt.plot(pred.data.numpy()[5, :, :], label='pred')
plt.plot(y_te[5, :, :], label='true')
plt.legend()


# %%

X = Variable(torch.from_numpy(x_te), requires_grad=True).float()
output = lstm(X)
pred = output[0].data.numpy()
one_hot = np.ones((pred.shape), dtype=np.float32)
one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
one_hot = torch.sum(one_hot*output[0])
gradient = grad(one_hot, X)[0].data.numpy()


# %%
plt.close()
idx = 8

plt.figure(3, figsize=(10, 6))
plt.subplot(3, 4, 1)
plt.bar(np.arange(0, 44, 1), gradient[idx, :, 0], color='r')
plt.title(data['VOInames'][1][0][0])
plt.subplot(3, 4, 2)
plt.bar(np.arange(0, 44, 1), gradient[idx, :, 1], color='b')
plt.title(data['VOInames'][2][0][0])
plt.subplot(3, 4, 3)
plt.bar(np.arange(0, 44, 1), gradient[idx, :, 2], color='g')
plt.title(data['VOInames'][3][0][0])
plt.subplot(3, 4, 4)
plt.bar(np.arange(0, 44, 1), gradient[idx, :, 3], color='y')
plt.title(data['VOInames'][4][0][0])
plt.subplot(3, 4, 5)
plt.bar(np.arange(0, 44, 1), gradient[idx, :, 4], color='c')
plt.title(data['VOInames'][5][0][0])
plt.subplot(3, 4, 6)
plt.bar(np.arange(0, 44, 1), gradient[idx, :, 5], color='m')
plt.title(data['VOInames'][6][0][0])
plt.subplot(3, 4, 7)
plt.bar(np.arange(0, 44, 1), gradient[idx, :, 6], color='k')
plt.title(data['VOInames'][7][0][0])
plt.subplot(3, 4, 8)
plt.bar(np.arange(0, 44, 1), gradient[idx, :, 7], color='pink')
plt.title(data['VOInames'][8][0][0])
plt.subplot(3, 4, 9)
plt.bar(np.arange(0, 44, 1), gradient[idx, :, 8], color='olive')
plt.title(data['VOInames'][9][0][0])
plt.subplot(3, 4, 10)
plt.bar(np.arange(0, 44, 1), gradient[idx, :, 9], color='gray')
plt.title(data['VOInames'][10][0][0])
plt.subplot(3, 4, 11)
plt.bar(np.arange(0, 44, 1), gradient[idx, :, 10], color='brown')
plt.title(data['VOInames'][11][0][0])
plt.tight_layout()
plt.subplot(3, 4, 12)
plt.plot(pred[idx], label='pred')
plt.plot(y_tr[idx], label='true')
plt.legend()
plt.show()

plt.figure(4, figsize=(10, 6))
plt.plot(x_tr[idx, :, 0], label=names[0], color='r')
plt.plot(x_tr[idx, :, 1], label=names[1], color='b')
plt.plot(x_tr[idx, :, 2], label=names[2], color='g')
plt.plot(x_tr[idx, :, 3], label=names[3], color='y')
plt.plot(x_tr[idx, :, 4], label=names[4], color='c')
plt.plot(x_tr[idx, :, 5], label=names[5], color='m')
plt.plot(x_tr[idx, :, 6], label=names[6], color='k')
plt.plot(x_tr[idx, :, 7], label=names[7], color='pink')
plt.plot(x_tr[idx, :, 8], label=names[8], color='olive')
plt.plot(x_tr[idx, :, 9], label=names[9], color='gray')
plt.plot(x_tr[idx, :, 10], label=names[10], color='brown')
plt.legend()
plt.show()
