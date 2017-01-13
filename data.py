import pdb
import numpy as np
import cPickle as pickle
import gzip

import draw


class Data_moving_mnist(object):
    def __init__(self, o):
        self.datafile = o.path_data+'/'+o.dataset+'/mnist.pkl.gz' 
        self.frmsz = o.moving_mnist['frmsz']
        self.featdim = o.moving_mnist['featdim']
        # TODO: change the feature dimension  
        # feature dimension will change once using CNN features. 
        # this will change the model too (input placeholder).
        self.outdim = o.moving_mnist['outdim']

        self.data = self._load_data()
        self.data_tr, self.data_va, self.data_te = self._split_data()
        assert self.data_tr['images'].shape[0]==self.data_tr['targets'].shape[0]
        assert self.data_va['images'].shape[0]==self.data_va['targets'].shape[0]
        assert self.data_te['images'].shape[0]==self.data_te['targets'].shape[0]
        self.ntr = self.data_tr['images'].shape[0]
        self.nva = self.data_va['images'].shape[0]
        self.nte = self.data_te['images'].shape[0]

        self.idx_shuffle_tr = np.random.permutation(self.ntr)
        self.idx_shuffle_va = np.random.permutation(self.nva)
        self.idx_shuffle_te = np.random.permutation(self.nte)

    def _load_data(self):
        with gzip.open(self.datafile, 'rb') as f:
             rawdata = pickle.load(f)
        data = {}
        for p, part in enumerate(('train', 'val', 'test')):
            data[part] = {}
            data[part]['images'] = rawdata[p][0].reshape(-1, 28, 28)
            data[part]['targets'] = rawdata[p][1]
        return data

    def _split_data(self):
        data_tr = self.data['train']
        data_va = self.data['val']
        data_te = self.data['test']
        return data_tr, data_va, data_te
             
    def load_batch(self, ib, o, data_=None):
        '''
        Everytime this function is called, create the batch number of moving 
        mnist sets. If this process has randomness, no other data augmentation 
        technique is applied for now. 
        '''
        if data_ == 'train':
            data = self.data_tr
            idx_shuffle = self.idx_shuffle_tr
        elif data_ == 'val':
            data = self.data_va
            idx_shuffle = self.idx_shuffle_va
        elif data_ == 'test':
            data = self.data_te
            idx_shuffle = self.idx_shuffle_te
        else:
            raise ValueError('wrong data')

        # the following is a modified version from RATM data preparation
        vids = np.zeros((o.batchsz, o.ntimesteps, self.frmsz, self.frmsz))
        pos_init = np.random.randint(self.frmsz-28, size=(o.batchsz,2))
        pos = np.zeros((o.batchsz, o.ntimesteps, 2), dtype=np.int32)
        pos[:,0] = pos_init

        posmax = self.frmsz-29

        d = np.random.randint(low=-15, high=15, size=(o.batchsz,2))
        idx = idx_shuffle[ib*o.batchsz:(ib+1)*o.batchsz]

        for t in range(o.ntimesteps):
            dtm1 = d
            d = np.random.randint(low=-15, high=15, size=(o.batchsz,2))
            for i in range(o.batchsz):
                vids[i,t,
                        pos[i,t,0]:pos[i,t,0]+28,
                        pos[i,t,1]:pos[i,t,1]+28] = \
                                data['images'][idx[i]]
            if t < o.ntimesteps-1:
                pos[:,t+1] = pos[:,t]+.1*d+.9*dtm1

                # check for proposer position (reflect if necessary)
                reflectidx = np.where(pos[:,t+1] > posmax)
                pos[:,t+1][reflectidx] = (posmax - 
                        (pos[:,t+1][reflectidx] % posmax))
                reflectidx = np.where(pos[:,t+1] < 0)
                pos[:,t+1][reflectidx] = -pos[:,t+1][reflectidx]

        # TODO: variable length inputs for
        # 1. online learning, 2. arbitrary training sequences 
        inputs_length = np.ones((o.batchsz), dtype=np.int32) * o.ntimesteps

        # TODO: feature might not need to be flattened. change accordingly.
        vids = vids.reshape((vids.shape[0],vids.shape[1],self.featdim))

        batch = {
                'inputs': vids,
                'inputs_length': inputs_length, 
                'labels': pos,
                'digits': data['targets'][idx]
                }
        return batch, idx

    def run_sanitycheck(self, batch):
        draw.show_moving_mnist(batch)


class Data_another_dataset(object):
    def __init__(self, o):
        print 'another dataset..'
        
    def load_data(self):
        print 'load raw dataset..'

    def load_batch(self):
        print 'loading batch'


def load_data(o):
    if o.dataset == 'moving_mnist':
        loader = Data_moving_mnist(o)
    elif o.dataset == 'another_dataset':
        raise ValueError('dataset not implemented yet')
    else:
        raise ValueError('wrong dataset')
    return loader

if __name__ == '__main__':
    print 'test dataset classes'

