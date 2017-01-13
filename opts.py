import pdb
import numpy as np
import os
import tensorflow as tf

import helpers


class Opts(object):
    def __init__(self):
        ''' Initial parameter settings 
        '''
        #----------------------------------------------------------------------
        # settings 
        self.mode               = None # 'train' or 'test'
        self.debugmode          = False # True or False
        self.seed_global        = 9
        self.dtype              = tf.float32

        #----------------------------------------------------------------------
        # data set 
        self.dataset            = 'moving_mnist' # (moving_mnist, etc.)
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
        self.batchsz            = 1
        self.optimizer          = 'sgd' # sgd, adam, rmsprop
        self.lr                 = 0.001
        self.lr_update          = False
        self.wd                 = 0.0
        self.grad_clip          = False
        self.max_grad_norm      = 5.0
        self.regularization     = False # weight regularization!

        #----------------------------------------------------------------------
        # save (save training results), load (test), resume (keep training)
        self.path_base          = os.path.dirname(__file__)
        self.path_data          = os.path.join(self.path_base, 'data')
        self.nosave             = False
        self.path_save          = os.path.join(self.path_base, 'save/'+helpers.get_time())
        self.path_model         = os.path.join(self.path_save, 'models')
        self.restore            = False 
        self.restore_model      = None # 'specify_pretrained_model.cpkt' 
        self.resume             = False
        self.resume_model       = None # 'specify_not_fully_trained_model.cpkt'
        # TODO: resume is on different purpose from restore and will need 
        # different stuff to be saved as dictionary

        #----------------------------------------------------------------------
        # memory allocation 
        self.gpu_manctrl        = False
        self.gpu_frac           = 0.5

    
    def update_by_sysarg(self, args):
        ''' Update options by system arguments
        '''
        print ''
        for ko, vo in vars(self).items():
            for ka, va in vars(args).items():
                if ko == ka and vo != va:
                    setattr(self, ko, va)
                    print 'opt changed ({}): {} -> {}'.format(ko, vo, va)
        print ''

    def initialize(self):
        '''
        Put initialize functions here; Make them as (pseudo) private
        '''
        if not self.nosave:
            self._create_save_directories()

        #self._print_settings()
        #self._save_settings()
        self._run_sanitycheck()

    def _create_save_directories(self):
        os.makedirs(self.path_save)
        os.makedirs(self.path_model)
    
    def _print_settings(self):
        '''Print current parameter settings 
        '''
        # TODO: implement this function
    
    def _save_settings(self):
        '''Save current parameter settings
        '''
        # TODO: implement this function

    def _run_sanitycheck(self):
        '''Options sanity check!
        '''
        # TODO: put more assertions!
        assert ((not self.restore and self.restore_model is None) or 
                (self.restore and self.restore_model is not None))
        assert ((not self.resume and self.resume_model is None) or 
                (self.resume and self.resume_model is not None))
        assert (
                (self.mode=='test' 
                    and self.restore and self.restore_model is not None) or 
                (self.mode=='train' 
                    and not self.restore and self.restore_model is None))


