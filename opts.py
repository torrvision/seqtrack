import pdb
import numpy as np
import os


class Opts(object):
    def __init__(self):
        ''' Initial parameter settings 
        '''
        #----------------------------------------------------------------------
        # basic settings 
        self.path_base          = os.path.dirname(__file__)
        self.path_data          = os.path.join(self.path_base, 'data')
        self.mode               = 'train' # 'train' or 'test'
        self.seed_general       = 9

        #----------------------------------------------------------------------
        # data set
        self.dataset            = 'moving_mnist' # options: 
        
        #----------------------------------------------------------------------
        # model parameters
        self.model              = 'rnn_basic'
        self.cell_type          = 'LSTM' 
        self.cell_hidden_size   = 300 # or dimension?
        self.ntimesteps         = 30
        self.nlayers            = 1
        self.dropout            = False
        self.keep_ratio         = 0.5

        #----------------------------------------------------------------------
        # training policies
        self.nepoch             = 10
        self.batchsz            = 1
        self.optimizer          = 'sgd' # sgd, adam, rmsprop
        self.lr                 = 0.001
        self.wd                 = 0.0
        self.grad_clip          = False
        self.max_grad_norm      = 5.0

        # test params
        self.testval            = 10
    
    def update_by_sysarg(self, args):
        ''' Update options by system arguments
        '''
        for ko, vo in vars(self).items():
            for ka, va in vars(args).items():
                if ko == ka and vo != va:
                    setattr(self, ko, va)
                    print 'opt changed ({}): {} -> {}'.format(ko, vo, va)

