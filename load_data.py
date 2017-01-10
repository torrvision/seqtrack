"""
Loads a raw dataset and pre-process to some degree. 
However, it but does not perform data augmentation. That is, data augmentation 
required during training will be performed later somewhere else. 
"""

import pdb
import numpy as np
import cPickle as pickle
import gzip


def load_data(o):
    # load and pre-process raw data
    if o.dataset == 'moving_mnist':
        with gzip.open(o.path_data+'/'+o.dataset+'/mnist.pkl.gz', 'rb') as f:
            rawdata = pickle.load(f)
        data = {}
        for p, part in enumerate(('train', 'val', 'test')):
            data[part] = {}
            data[part]['images'] = rawdata[p][0].reshape(-1, 28, 28)
            data[part]['targets'] = rawdata[p][1]
    else:
        raise ValueError('dataset not implemented yet or simply wrong..')

    # split data into train, validation, test
    return split_data(data, o)

def split_data(data, o):
    if o.dataset == 'moving_mnist':
        data_tr = data['train']
        data_va = data['val']
        data_te = data['test']
        return data_tr, data_va, data_te
    else:
        raise ValueError('dataset not implemented yet or simply wrong..')

    

