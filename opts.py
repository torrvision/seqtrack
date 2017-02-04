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
        self.frmsz              = None
        self.ninchannel         = None
        self.outdim             = None

        #----------------------------------------------------------------------
        # model - general
        self.model              = 'rnn_attention_st' # {rnn_basic, rnn_attention_s, rnn_attention_t}
        self.usetfapi           = False

        #----------------------------------------------------------------------
        # model - attention
        self.h_concat_ratio     = 1

        #----------------------------------------------------------------------
        # model parameters - rnn
        self.cell_type          = 'LSTM' 
        self.nunits             = 300 
        self.ntimesteps         = 30
        self.rnn_nlayers        = 1
        self.dropout_rnn        = False
        self.keep_ratio         = 0.5
        self.lstmforgetbias     = False
        self.yprev_mode         = '' # nouse, concat_abs, concat_spatial (later), weight

        #----------------------------------------------------------------------
        # model parameters - cnn (or feature extractor)
        self.cnn_pretrain       = False 
        self.cnn_model          = 'vgg' # vgg, resnet, imagenet, etc.
        self.cnn_nchannels      = [16, 16]
        self.cnn_nlayers        = 2
        self.cnn_filtsz         = [3, 3]
        self.cnn_strides        = [3, 3]
        # TODO: add dropout and batch norm option

        #----------------------------------------------------------------------
        # training policies
        self.nepoch             = 10
        self.batchsz            = 1
        self.optimizer          = 'sgd' # sgd, adam, rmsprop
        self.lr                 = 0.001
        self.lr_update          = False
        self.wd                 = 0.0 # weight decay for regularization
        self.grad_clip          = False
        self.max_grad_norm      = 5.0

        #----------------------------------------------------------------------
        # save (save training results), load (test), resume (keep training)
        self.path_base          = os.path.dirname(__file__)
        self.path_data          = os.path.join(self.path_base, 'data')
        self.nosave             = False
        self.path_save          = os.path.join(
                                    self.path_base, 'save/'+self.exectime)
        #self.path_save_tmp      = os.path.join(
                                    #self.path_base, 'tmp/'+self.exectime)
        self.path_save_tmp      = os.path.join(self.path_base, 'tmp/') #TODO:tmp
        self.path_model         = os.path.join(self.path_save, 'models')
        self.path_loss          = os.path.join(self.path_save, 'losses')
        self.path_eval          = os.path.join(self.path_save, 'evals')
        self.restore            = False 
        self.restore_model      = None # 'specify_pretrained_model.cpkt' 
        self.resume             = False
        self.resume_data        = None

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
        self._run_sanitycheck()
        self._create_save_directories()
        self._set_gpu_config()
        #self._print_settings()
        #self._save_settings()
        self._set_dataset_params()

    def _set_dataset_params(self):
        if self.dataset == 'moving_mnist' or self.dataset == 'bouncing_mnist':
            self.frmsz = 100 # image size (assuming square)
            self.ninchannel = 1 # number of image channels
            self.outdim = 4 # rnn final output
        else:
            raise ValueError('not implemented yet, coming soon..')

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
        assert((not self.resume and self.resume_data is None) or
                (self.resume and self.resume_data is not None))
        assert(self.cnn_nlayers == len(self.cnn_nchannels))
        assert(self.cnn_nlayers == len(self.cnn_filtsz))
        assert(self.cnn_nlayers == len(self.cnn_strides))

    def _create_save_directories(self):
        if not self.nosave:
            #os.makedirs(self.path_save)
            helpers.mkdir_p(self.path_save)
            if self.mode == 'train':
                os.makedirs(self.path_model)
                os.makedirs(self.path_loss)
            elif self.mode == 'test':
                os.makedirs(self.path_eval)
            else:
                raise ValueError('currently mode should be only train or test')

    def _print_settings(self):
        '''Print current parameter settings 
        '''
        # TODO: implement this function
    
    def _save_settings(self):
        '''Save current parameter settings
        '''
        # TODO: implement this function



if __name__ == '__main__':
    '''Test options
    '''

    o = Opts()
    pdb.set_trace()
