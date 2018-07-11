import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 n_layers,
                 batch_size,
                 bidirectional):
        super(LSTM, self).__init__()

        self.n_layers = n_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        if bidirectional:
            self.hidden_layer = nn.LSTM(input_size,
                                        hidden_size,
                                        n_layers,
                                        dropout=0.05,
                                        bidirectional=True,
                                        batch_first=True)
            self.output_layer = nn.Linear(2*hidden_size, output_size)
        else:
            self.hidden_layer = nn.LSTM(input_size,
                                        hidden_size,
                                        n_layers,
                                        dropout=0.05,
                                        bidirectional=False,
                                        batch_first=True)
            self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        hidden, cell = self.initHidden(), self.initHidden()
        outputs = []
        for i in range(inp.size(1)):
            rnn_out, hidden = self.hidden_layer(inp[:, i:i+1, :], (hidden, cell))
            outputs.append(self.output_layer(rnn_out.squeeze()))
            hidden, cell = hidden[0], hidden[1]
        return torch.stack(outputs, dim=1), hidden

    def pred(self, inp):
        out = self.forward(inp)
        return out[0]

    def initHidden(self):
        if self.bidirectional:
            out = Variable(torch.zeros(2*self.n_layers,
                                       self.batch_size,
                                       self.hidden_size))
        else:
            out = Variable(torch.zeros(self.n_layers,
                                       self.batch_size,
                                       self.hidden_size))
        return out

    def batch(self, x_tr, y_tr):
        index = np.random.permutation(x_tr.size()[0])
        return x_tr[index, :, :], y_tr[index, :, :]

