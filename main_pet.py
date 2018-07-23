# %%
import torch
import pandas
import datetime
import argparse
import torch.nn as nn
from GRU_rnn import GRU
from Elman_rnn import RNN
from LSTM_rnn import LSTM
from utility import train, validate, test, Early_Stopper
from PET_graph_loader import proper_split,  proper_split_LV

n_layers = 2
n_hidden = 10
n_outputs = 1
patience = 1
n_epochs = 0
rnn_loss = []
gru_loss = []
lstm_loss = []
batch_size = 12
bidirectional = False
model_selector_rnn = Early_Stopper(patience)
model_selector_gru = Early_Stopper(patience)
model_selector_lstm = Early_Stopper(patience)
criterion = nn.MSELoss()

parser = argparse.ArgumentParser()
parser.add_argument("file")
parser.add_argument("data_split")
parser.add_argument('--cuda', action='store_true', default=False)


args = parser.parse_args()

if args.data_splot == 0:
    x_tr, y_tr, x_va, y_va, x_te, y_te = proper_split(parser.file)
    n_inputs = 11
if args.data_splot == 1:
    x_tr, y_tr, x_va, y_va, x_te, y_te = proper_split_LV(parser.file)
    n_inputs = 10
if args.data_splot == 2:
    x_tr, y_tr, x_va, y_va, x_te, y_te = proper_split(parser.file)


print('Data loading complete')

if args.cuda:
    rnn = RNN(n_inputs, n_hidden, n_outputs, n_layers, batch_size, bidirectional).cuda()
    gru = GRU(n_inputs, n_hidden, n_outputs, n_layers, batch_size, bidirectional).cuda()
    lstm = LSTM(n_inputs, n_hidden, n_outputs, n_layers, batch_size, bidirectional).cuda()
else:
    rnn = RNN(n_inputs, n_hidden, n_outputs, n_layers, batch_size, bidirectional)
    gru = GRU(n_inputs, n_hidden, n_outputs, n_layers, batch_size, bidirectional)
    lstm = LSTM(n_inputs, n_hidden, n_outputs, n_layers, batch_size, bidirectional)

optimizer_rnn = torch.optim.Adam(rnn.parameters())
optimizer_gru = torch.optim.Adam(gru.parameters())
optimizer_lstm = torch.optim.Adam(lstm.parameters())

time_start = datetime.datetime.now()
print('Training started at:', time_start)

while(model_selector_rnn.keep_training or
      model_selector_gru.keep_training or
      model_selector_lstm.keep_training):

    if model_selector_rnn.keep_training:
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

    if model_selector_gru.keep_training:
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

print('Training ended')

s1 = pandas.Series([model_selector_rnn.final_epoch, rnn_loss[-patience-1][0],
                    rnn_loss[-patience-1][1], rnn_loss[-patience-1][2],
                    rnn_time])
s2 = pandas.Series([model_selector_gru.final_epoch, gru_loss[-patience-1][0],
                    gru_loss[-patience-1][1], gru_loss[-patience-1][2],
                    gru_time])
s3 = pandas.Series([model_selector_lstm.final_epoch, lstm_loss[-patience-1][0],
                    lstm_loss[-patience-1][1], lstm_loss[-patience-1][2],
                    gru_time])

print(pandas.DataFrame([list(s1), list(s2), list(s3)],
                       index=['RNN', 'GRU', 'LSTM'],
                       columns=['Final Epoch',
                                'Training',
                                'Validation',
                                'Test',
                                'Elapsed time']))
