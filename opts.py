import pdb
import numpy as np
import os
import tensorflow as tf


class Opts(object):
    def __init__(self):
        ''' Initial parameter settings 
        '''
        #----------------------------------------------------------------------
        # basic settings 
        self.path_base          = os.path.dirname(__file__)
        self.path_data          = os.path.join(self.path_base, 'data')
        self.mode               = '' # 'train' or 'test'
        self.seed_general       = 9
        self.dtype              = tf.float32

        #----------------------------------------------------------------------
        # data set 
        self.dataset            = 'moving_mnist' # (moving_mnist, , ,)
        self.moving_mnist       = {
                                    'frmsz': 100,
                                    'featdim': 100*100,
                                    'outdim': 2 
                                    }

        
        #----------------------------------------------------------------------
        # model parameters - rnn
        self.model              = 'rnn_basic'
        self.cell_type          = 'LSTM' 
        self.nunits             = 300 # or dimension?
        self.ntimesteps         = 30
        self.nlayers            = 1
        self.dropout_rnn        = False
        self.keep_ratio         = 0.5

        #----------------------------------------------------------------------
        # model parameters - cnn (or feature extractor)
        self.model_cnn          = 'vgg' # vgg, resnet, imagenet, etc.

        #----------------------------------------------------------------------
        # training policies
        self.nepoch             = 10
        self.batchsz            = 16
        self.optimizer          = 'sgd' # sgd, adam, rmsprop
        self.lr                 = 0.001
        self.lr_update          = False
        self.wd                 = 0.0
        self.grad_clip          = False
        self.max_grad_norm      = 5.0
        self.regularization     = False # weight regularization!

    
    def update_by_sysarg(self, args):
        ''' Update options by system arguments
        '''
        for ko, vo in vars(self).items():
            for ka, va in vars(args).items():
                if ko == ka and vo != va:
                    setattr(self, ko, va)
                    print 'opt changed ({}): {} -> {}'.format(ko, vo, va)

