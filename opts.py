import pdb
import numpy as np
import os
import tensorflow as tf
import socket
import random

import helpers


class Opts(object):
    def __init__(self):
        ''' Initial parameter settings 
        '''
        #----------------------------------------------------------------------
        # settings 
        self.mode               = '' # 'train' or 'test'
        self.debugmode          = False # True or False
        self.seed_global        = 9
        self.dtype              = tf.float32
        self.exectime           = helpers.get_time()
        self.tfversion          = tf.__version__[0:4]

        #----------------------------------------------------------------------
        # data set specific parameters 
        # TODO: only params that need to change; otherwise put it in data class
        self.dataset            = '' # (bouncing_mnist, etc.)
        self.trainsplit         = None # 0,1,2,3 or 9 for all train sets
        self.frmsz              = None
        self.ninchannel         = None
        self.outdim             = None
        self.useresizedimg      = True

        #----------------------------------------------------------------------
        # model - general
        self.model              = '' # {rnn_basic, rnn_attention_s, rnn_attention_t}
        self.losses             = None

        #----------------------------------------------------------------------
        # model - attention
        self.h_concat_ratio     = 1

        #----------------------------------------------------------------------
        # model parameters - rnn
        self.cell_type          = 'LSTM' 
        self.nunits             = 512
        self.ntimesteps         = 20
        self.rnn_nlayers        = 1
        self.dropout_rnn        = False
        self.keep_ratio         = 0.5
        self.lstmforgetbias     = False
        self.yprev_mode         = '' # nouse, concat_abs, concat_spatial (later), weight
        self.pass_yinit         = False

        #----------------------------------------------------------------------
        # model parameters - cnn (or feature extractor)
        self.cnn_pretrain       = False 
        self.cnn_model          = 'vgg' # vgg, resnet, imagenet, etc.
        self.dropout_cnn        = False
        self.keep_ratio_cnn     = 0.1
        # TODO: add dropout and batch norm option

        #----------------------------------------------------------------------
        # model parameters - NTM
        

        #----------------------------------------------------------------------
        # training policies
        self.nepoch             = 20
        self.batchsz            = 1
        self.optimizer          = 'adam' # sgd, adam, rmsprop
        self.lr                 = 0.0001
        self.lr_update          = False
        self.wd                 = 0.0 # weight decay for regularization
        self.grad_clip          = False
        self.max_grad_norm      = 5.0

        #----------------------------------------------------------------------
        # save (save training results), load (test), resume (keep training)
        self.path_src           = os.path.dirname(__file__) # Take aux/ and stat/ from source dir.
        self.path_aux           = os.path.join(self.path_src, 'aux')
        self.path_stat          = os.path.join(self.path_src, 'stat')
        self.path_data_home     = './data'
        self.path_data          = '' # This is set later e.g. {path_data_home}/ILSVRC
        self.nosave             = False
        self.path_ckpt          = './ckpt'
        self.period_ckpt        = 10000
        self.period_assess      = self.period_ckpt
        self.path_output        = './output'
        self.restore            = False 
        self.restore_model      = None # 'specify_pretrained_model.cpkt' 
        self.resume             = False
        self.resume_data        = None
        #self.path_logs          = './logs'
        # 'logs' directory reserved for saving system level logs (being used in
        # a shell script)
        self.path_summary       = './summary'
        self.summary_period     = 10
        self.val_period         = 10

        #----------------------------------------------------------------------
        # custom libraries
        self.path_home          = os.path.expanduser('~')
        self.path_customlib     = os.path.join(self.path_home, 
                                'tensorflow/bazel-bin/tensorflow/core/user_ops') 

        #----------------------------------------------------------------------
        # device memory allocation 
        self.device             = 'gpu' # cpu or gpu; Don't change it for now
        self.device_number      = 0 # TODO: not working on local machine
        self.gpu_manctrl        = False
        self.gpu_frac           = 0.4 # TODO: is this optimal value? 

    
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
        '''Put initialize functions here; Make them as (pseudo) private
        '''
        tf.set_random_seed(self.seed_global) # TODO: not 100% certain
        np.random.seed(self.seed_global) # checked! 
        random.seed(self.seed_global)
        self._run_sanitycheck()
        self._set_gpu_config()
        self._set_dataset_params()

    def _set_dataset_params(self):
        if self.dataset in ['moving_mnist', 'bouncing_mnist']:
            self.frmsz = 100 # image size (assuming square)
            self.ninchannel = 1 # number of image channels
            self.outdim = 4 # rnn final output
        elif self.dataset in ['ILSVRC', 'OTB-50', 'OTB-100']:
            # TODO: try different frmsz
            if not self.frmsz:
                self.frmsz = 100 # image (re)size, width and height. assuming square
            self.ninchannel = 3
            self.outdim = 4 # rnn final output
        else:
            raise ValueError('not implemented yet, coming soon..')
        self.path_data = os.path.join(self.path_data_home, self.dataset)

    def _set_gpu_config(self):
        self.tfconfig = tf.ConfigProto()
        # TODO: not sure if this should be always true.
        self.tfconfig.allow_soft_placement = True
        #self.tfconfig.log_device_placement = True
        if self.gpu_manctrl:
            self.tfconfig.gpu_options.allow_growth = True
            self.tfconfig.gpu_options.per_process_gpu_memory_fraction \
                    = self.gpu_frac

    def _run_sanitycheck(self):
        '''Options sanity check!
        '''
        assert(self.mode == 'train' or self.mode == 'test')
        assert((not self.restore and self.restore_model is None) or 
                (self.restore and self.restore_model is not None))
        assert(
                (self.mode=='test' 
                    and self.restore and self.restore_model is not None) or 
                (self.mode=='train' 
                    and not self.restore and self.restore_model is None))


if __name__ == '__main__':
    '''Test options
    '''
    o = Opts()
    pdb.set_trace()
