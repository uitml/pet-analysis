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
n_hidden = 20
n_outputs = 1
patience = 50
batch_size = 12
bidirectional = False
criterion = nn.MSELoss()

parser = argparse.ArgumentParser()
parser.add_argument("file")
parser.add_argument("data_split")
parser.add_argument('--cuda', action='store_true', default=False)

args = parser.parse_args()

for i in range(3):

    print('Random suffle: ', i)

    if args.data_split == 'VC':
        x_tr, y_tr, x_va, y_va, x_te, y_te = proper_split(args.file)
        n_inputs = 11
    if args.data_split == 'LV':
        x_tr, y_tr, x_va, y_va, x_te, y_te = proper_split_LV(args.file)
        n_inputs = 10
    if args.data_split == 2:
        x_tr, y_tr, x_va, y_va, x_te, y_te = proper_split(args.file)
    
    
    print('Data loading complete')
    
    if args.cuda:
        rnn = RNN(n_inputs, n_hidden, n_outputs,
                  n_layers, batch_size, bidirectional, args.cuda).cuda()
        gru = GRU(n_inputs, n_hidden, n_outputs,
                  n_layers, batch_size, bidirectional, args.cuda).cuda()
        lstm = LSTM(n_inputs, n_hidden, n_outputs,
                    n_layers, batch_size, bidirectional, args.cuda).cuda()
    else:
        rnn = RNN(n_inputs, n_hidden, n_outputs,
                  n_layers, batch_size, bidirectional, args.cuda)
        gru = GRU(n_inputs, n_hidden, n_outputs,
                  n_layers, batch_size, bidirectional, args.cuda)
        lstm = LSTM(n_inputs, n_hidden, n_outputs,
                    n_layers, batch_size, bidirectional, args.cuda)

    n_epochs = 0
    rnn_loss = []
    gru_loss = []
    lstm_loss = []

    optimizer_rnn = torch.optim.Adam(rnn.parameters())
    optimizer_gru = torch.optim.Adam(gru.parameters())
    optimizer_lstm = torch.optim.Adam(lstm.parameters())

    model_selector_rnn = Early_Stopper(patience)
    model_selector_gru = Early_Stopper(patience)
    model_selector_lstm = Early_Stopper(patience)

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
                                   args.cuda),
                            validate(x_va,
                                     y_va,
                                     batch_size,
                                     criterion,
                                     rnn,
                                     args.cuda),
                            test(x_te,
                                 y_te,
                                 batch_size,
                                 criterion,
                                 rnn,
                                 args.cuda)])
    
            rnn_time = str(datetime.datetime.now()-time_start)
            model_selector_rnn.update(rnn_loss[-1][1], n_epochs)
    
        if model_selector_gru.keep_training:
            gru_loss.append([train(x_tr,
                                   y_tr,
                                   batch_size,
                                   optimizer_gru,
                                   criterion,
                                   gru,
                                   args.cuda),
                            validate(x_va,
                                     y_va,
                                     batch_size,
                                     criterion,
                                     gru,
                                     args.cuda),
                            test(x_te,
                                 y_te,
                                 batch_size,
                                 criterion,
                                 gru,
                                 args.cuda)])
    
            gru_time = str(datetime.datetime.now()-time_start)
            model_selector_gru.update(gru_loss[-1][1], n_epochs)
    
        if model_selector_lstm.keep_training:
            lstm_loss.append([train(x_tr,
                                    y_tr,
                                    batch_size,
                                    optimizer_lstm,
                                    criterion,
                                    lstm,
                                    args.cuda),
                             validate(x_va,
                                      y_va,
                                      batch_size,
                                      criterion,
                                      lstm,
                                      args.cuda),
                             test(x_te,
                                  y_te,
                                  batch_size,
                                  criterion,
                                  lstm,
                                  args.cuda)])

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

    s1 = pandas.Series([model_selector_rnn.final_epoch,
                        rnn_loss[-patience-1][0], rnn_loss[-patience-1][1],
                        rnn_loss[-patience-1][2], rnn_time])
    s2 = pandas.Series([model_selector_gru.final_epoch,
                        gru_loss[-patience-1][0], gru_loss[-patience-1][1],
                        gru_loss[-patience-1][2], gru_time])
    s3 = pandas.Series([model_selector_lstm.final_epoch,
                        lstm_loss[-patience-1][0], lstm_loss[-patience-1][1],
                        lstm_loss[-patience-1][2],
                        lstm_time])

    print(pandas.DataFrame([list(s1), list(s2), list(s3)],
                           index=['RNN', 'GRU', 'LSTM'],
                           columns=['Final Epoch',
                                    'Training',
                                    'Validation',
                                    'Test',
                                    'Elapsed time']))
