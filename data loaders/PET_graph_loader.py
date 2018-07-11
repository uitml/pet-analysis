# %%
import numpy as np
import scipy.io as sio

##############################################################################
# Script for importing graphs created from PET-images of mice.
# Data is returend in a shape suitable for RNNs in Pytorch.
##############################################################################


def data_loader(path):
    # Function for importing data in raw form.

    # path: Path to stored data.
    return sio.loadmat(path)['D']


def leave_one_out(path):
    # Function which splits data into training and test set
    # according to the "leave one out" method, that is, training on all
    # samples except one which we use for testing.

    data = data_loader(path)                        # Import data.
    data = np.concatenate((data[:, :, :, 0:23], data[:, :, :, 24:]), 3) # Remoev NaN
    idx_tr = np.random.permutation(data.shape[-1])  # Index for data shuffling.
    print('Mouse number {}'.format(idx_tr[-1]), 'is test mouse')

    # Extract training data and training labels
    # Reshaping into form (batch_size, time, variables).
    x_tr = data[:, 4, 1:, idx_tr[:-1]]
    y_tr = data[:, 4, 0, idx_tr[:-1]]
    y_tr = np.transpose(y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1), (1, 0, 2))

    x_te = data[:, 4, 1:, idx_tr[-1]]
    y_te = data[:, 4, 0, idx_tr[-1]]
    y_te = np.transpose(y_te.reshape(y_te.shape[0], 1, 1), (1, 0, 2))

    return (np.asarray(x_tr, dtype=np.float32),
            np.asarray(y_tr, dtype=np.float32),
            np.asarray(x_te, dtype=np.float32),
            np.asarray(y_te, dtype=np.float32))


def proper_split(path):

    # Function which splits data into training, validation and test set.
    # We use 60% of the data for training and 20% for validation and testing.

    data = data_loader(path)                        # Import data.
    data = np.concatenate((data[:, :, :, 0:23], data[:, :, :, 24:]), 3) # Remoev NaN
    idx = np.random.permutation(data.shape[-1])     # Index for data shuffling.
    print('Splitting data intro 22 training mice,'
          '\n 12 validation mice and 12 test mice.')

    # Extract training, validation and test data/labels.
    # Reshaping into form (time, batch size, variables), suitable for Pytorch.
    x_tr = data[:, 4, 1:, idx[0:44]]
    y_tr = data[:, 4, 0, idx[0:44]]
    y_tr = np.transpose(y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1), (1, 0, 2))

    x_va = data[:, 4, 1:, idx[44:56]]
    y_va = data[:, 4, 0, idx[44:56]]
    y_va = np.transpose(y_va.reshape(y_va.shape[0], y_va.shape[1], 1), (1, 0, 2))

    x_te = data[:, 4, 1:, idx[56:68]]
    y_te = data[:, 4, 0, idx[56:68]]
    y_te = np.transpose(y_te.reshape(y_te.shape[0], y_te.shape[1], 1), (1, 0, 2))

    return (np.asarray(x_tr, dtype=np.float32),
            np.asarray(y_tr, dtype=np.float32),
            np.asarray(x_va, dtype=np.float32),
            np.asarray(y_va, dtype=np.float32),
            np.asarray(x_te, dtype=np.float32),
            np.asarray(y_te, dtype=np.float32))
