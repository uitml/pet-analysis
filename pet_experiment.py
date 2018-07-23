# %%
import torch
import pandas
import datetime
import numpy as np
import torch.nn as nn
import scipy.io as sio
from GRU_rnn import GRU
from Elman_rnn import RNN
from LSTM_rnn import LSTM
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utility import train, validate, test, Early_Stopper
from PET_graph_loader import proper_split, leave_one_out, proper_split_LV

path = '/home/kristoffer/scripts/data/PET_voistat_data_12_VOIs_n68_v7.mat'
#x_tr, y_tr, x_te, y_te = leave_one_out(path)
#x_tr, y_tr, x_va, y_va, x_te, y_te = proper_split(path)
x_tr, y_tr, x_va, y_va, x_te, y_te = proper_split_LV(path)
data = sio.loadmat(path)
names = []
for i in range(11):
    names.append(data['VOInames'][i+1][0][0])

# %%

n_inputs = 10
n_layers = 2
n_hidden = 10
n_outputs = 1
batch_size = 12
bidirectional = False
time_start = datetime.datetime.now()

rnn = RNN(n_inputs, n_hidden, n_outputs, n_layers, batch_size, bidirectional)
gru = GRU(n_inputs, n_hidden, n_outputs, n_layers, batch_size, bidirectional)
lstm = LSTM(n_inputs, n_hidden, n_outputs, n_layers, batch_size, bidirectional)

patience = 2
n_epochs = 0
rnn_loss = []
gru_loss = []
lstm_loss = []
model_selector_rnn = Early_Stopper(patience)
model_selector_gru = Early_Stopper(patience)
model_selector_lstm = Early_Stopper(patience)
criterion = nn.MSELoss()
optimizer_rnn = torch.optim.Adam(rnn.parameters())
optimizer_gru = torch.optim.Adam(gru.parameters())
optimizer_lstm = torch.optim.Adam(lstm.parameters())

print('Training started at:', time_start)

while(model_selector_rnn.keep_training or
      model_selector_gru.keep_training or
      model_selector_lstm.keep_training):

    if model_selector_rnn:
        rnn_loss.append([train(x_tr,
                               y_tr,
                               batch_size,
                               optimizer_rnn,
                               criterion,
                               rnn,
                               False),
                        validate(x_va,
                                 y_va,
                                 batch_size,
                                 criterion,
                                 rnn,
                                 False),
                        test(x_te,
                             y_te,
                             batch_size,
                             criterion,
                             rnn,
                             False)])

        rnn_time = str(datetime.datetime.now()-time_start)
        model_selector_rnn.update(rnn_loss[-1][1], n_epochs)

    if model_selector_gru:
        gru_loss.append([train(x_tr,
                               y_tr,
                               batch_size,
                               optimizer_gru,
                               criterion,
                               gru,
                               False),
                        validate(x_va,
                                 y_va,
                                 batch_size,
                                 criterion,
                                 gru,
                                 False),
                        test(x_te,
                             y_te,
                             batch_size,
                             criterion,
                             gru,
                             False)])

        gru_time = str(datetime.datetime.now()-time_start)
        model_selector_rnn.update(gru_loss[-1][1], n_epochs)

    if model_selector_lstm.keep_training:
        lstm_loss.append([train(x_tr,
                                y_tr,
                                batch_size,
                                optimizer_lstm,
                                criterion,
                                lstm,
                                False),
                         validate(x_va,
                                  y_va,
                                  batch_size,
                                  criterion,
                                  lstm,
                                  False),
                         test(x_te,
                              y_te,
                              batch_size,
                              criterion,
                              lstm,
                              False)])

        lstm_time = str(datetime.datetime.now()-time_start)
        model_selector_lstm.update(lstm_loss[-1][1], n_epochs)

    n_epochs += 1

    s1 = pandas.Series([n_epochs, rnn_loss[-1][0], rnn_loss[-1][1],
                        rnn_loss[-1][2], rnn_time])
    s2 = pandas.Series([n_epochs, gru_loss[-1][0], gru_loss[-1][1],
                        gru_loss[-1][2], gru_time])
    s3 = pandas.Series([n_epochs, lstm_loss[-1][0], lstm_loss[-1][1],
                        lstm_loss[-1][2], lstm_time])

    print(pandas.DataFrame([list(s1), list(s2), list(s3)],
                           index=['RNN', 'GRU', 'LSTM'],
                           columns=['Epoch',
                                    'Training',
                                    'Validation',
                                    'Test',
                                    'Elapsed time']))


# %%

plt.figure(9)
plt.subplot(1, 2, 1)
plt.plot([item[0] for item in rnn_loss], label='RNN')
plt.plot([item[0] for item in gru_loss], label='GRU')
plt.plot([item[0] for item in lstm_loss], label='LSTM')
plt.legend()
plt.title('Train')
plt.subplot(1, 2, 2)
plt.plot([item[1] for item in rnn_loss], label='RNN')
plt.plot([item[1] for item in gru_loss], label='GRU')
plt.plot([item[1] for item in lstm_loss], label='LSTM')
plt.legend()
plt.title('Validation')


# %%

idx = np.random.permutation(12)

pred_rnn = rnn.pred(Variable(torch.from_numpy(x_te)))
pred_gru = gru.pred(Variable(torch.from_numpy(x_te)))
pred_lstm = lstm.pred(Variable(torch.from_numpy(x_te)))

plt.figure(1, figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.plot(pred_rnn.data.numpy()[idx[0], :, :], label='RNN')
plt.plot(pred_gru.data.numpy()[idx[0], :, :], label='GRU')
plt.plot(pred_lstm.data.numpy()[idx[0], :, :], label='LSTM')
plt.plot(y_te[0, :, :], label='true')
plt.legend()
plt.title(idx[0])
plt.subplot(2, 3, 2)
plt.plot(pred_rnn.data.numpy()[idx[1], :, :], label='RNN')
plt.plot(pred_gru.data.numpy()[idx[1], :, :], label='GRU')
plt.plot(pred_lstm.data.numpy()[idx[1], :, :], label='LSTM')
plt.plot(y_te[1, :, :], label='true')
plt.legend()
plt.title(idx[1])
plt.subplot(2, 3, 3)
plt.plot(pred_rnn.data.numpy()[idx[2], :, :], label='RNN')
plt.plot(pred_gru.data.numpy()[idx[2], :, :], label='GRU')
plt.plot(pred_lstm.data.numpy()[idx[2], :, :], label='LSTM')
plt.plot(y_te[2, :, :], label='true')
plt.legend()
plt.title(idx[2])
plt.subplot(2, 3, 4)
plt.plot(pred_rnn.data.numpy()[idx[3], :, :], label='RNN')
plt.plot(pred_gru.data.numpy()[idx[3], :, :], label='GRU')
plt.plot(pred_lstm.data.numpy()[idx[3], :, :], label='LSTM')
plt.plot(y_te[3, :, :], label='true')
plt.legend()
plt.title(idx[3])
plt.subplot(2, 3, 5)
plt.plot(pred_rnn.data.numpy()[idx[3], :, :], label='RNN')
plt.plot(pred_gru.data.numpy()[idx[3], :, :], label='GRU')
plt.plot(pred_lstm.data.numpy()[idx[3], :, :], label='LSTM')
plt.plot(y_te[4, :, :], label='true')
plt.legend()
plt.title(idx[4])
plt.subplot(2, 3, 6)
plt.plot(pred_rnn.data.numpy()[idx[4], :, :], label='RNN')
plt.plot(pred_gru.data.numpy()[idx[4], :, :], label='GRU')
plt.plot(pred_lstm.data.numpy()[idx[4], :, :], label='LSTM')
plt.plot(y_te[5, :, :], label='true')
plt.legend()
plt.title(idx[5])
plt.tight_layout()


# %%
import numpy as np
from torch.autograd import grad

X = Variable(torch.from_numpy(x_te), requires_grad=True).float()
output = rnn(X)
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

