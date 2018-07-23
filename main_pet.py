# %%
import torch
import pandas
import datetime
import argparse
import torch.nn as nn
from Elman_rnn import RNN
from utility import train
import matplotlib.pyplot as plt
from PET_graph_loader import leave_one_out


parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("file")
parser.add_argument('--cuda', action='store_true', default=False)

path = '/home/kristoffer/scripts/data/PET_voistat_data_12_VOIs_n68_v7.mat'
x_tr, y_tr, x_te, y_te = leave_one_out(path)