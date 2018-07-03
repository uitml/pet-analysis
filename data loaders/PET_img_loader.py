import hdf5storage
import numpy as np

##############################################################################
# Script for importing images created from PET-scanning of mice.
# Data is returend in a shape suitable for CNNs in Pytorch.
##############################################################################


def data_loader(path):
    # Function for importing data in raw form.

    # path: Path to stored data.
    return hdf5storage.loadmat('PET_image_data_12_VOIs_n68_v7-3.mat')['I']
