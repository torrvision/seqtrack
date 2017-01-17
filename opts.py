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

        #----------------------------------------------------------------------
        # data set 
        self.dataset            = 'moving_mnist' # (bouncing_mnist, etc.)
        self.moving_mnist       = {
                                    'frmsz': 100,
                                    'featdim': 100*100,
                                    'outdim': 4 
                                    }
        # TODO: if a param doesn't need to change, 
        # let's just fix it with a default, instead of making it optional
        # TODO: perhaps moving mnist should be the same
        self.bouncing_mnist     = {
                                    'frmsz': 100,
                                    'featdim': 10000,
                                    'outdim': 4
                                    #'nr_objs': 1,
                                    #'img_row': 100,
                                    #'img_col': 100,
                                    #'clutter_move': 1,
                                    #'with_clutters': 1,
                                    #'nr_objs': 1,
                                    #'seq_len': 20,
                                    #'acc_scale': 0.1,
                                    #'zoom_scale': 0.1,
                                    #'double_mnist': False 
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
        self.nepoch             = 1
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
        self.path_save          = os.path.join(
                                    self.path_base, 'save/'+self.exectime)
        self.path_save_tmp      = os.path.join(
                                    self.path_base, 'tmp/'+self.exectime)
        self.path_model         = os.path.join(self.path_save, 'models')
        self.path_eval          = os.path.join(self.path_save, 'evals')
        self.restore            = False 
        self.restore_model      = None # 'specify_pretrained_model.cpkt' 
        self.resume             = False
        self.resume_model       = None # 'specify_not_fully_trained_model.cpkt'
        # TODO: resume is on different purpose from restore and will need 
        # different stuff to be saved (as dict)

        #----------------------------------------------------------------------
        # device memory allocation 
        self.device             = 'gpu' # cpu or gpu; Don't change it for now
        self.device_number      = 0
        # TODO: Not working now. will figure out on different environment
        self.gpu_manctrl        = False
        self.gpu_frac           = 0.4 # TODO: is this optimal? 

    
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

    def _set_gpu_config(self):
        self.tfconfig = tf.ConfigProto()
        # TODO: not sure if this should be always true.
        self.tfconfig.allow_soft_placement = True
        if self.gpu_manctrl:
            self.tfconfig.gpu_options.allow_growth = True
            self.tfconfig.gpu_options.per_process_gpu_memory_fraction \
                    = o.gpu_frac

    def _run_sanitycheck(self):
        '''Options sanity check!
        '''
        # TODO: put more assertions!
        assert(self.mode == 'train' or self.mode == 'test')
        assert((not self.restore and self.restore_model is None) or 
                (self.restore and self.restore_model is not None))
        assert((not self.resume and self.resume_model is None) or 
                (self.resume and self.resume_model is not None))
        assert(
                (self.mode=='test' 
                    and self.restore and self.restore_model is not None) or 
                (self.mode=='train' 
                    and not self.restore and self.restore_model is None))

    def _create_save_directories(self):
        if not self.nosave:
            #os.makedirs(self.path_save)
            helpers.mkdir_p(self.path_save)
            if self.mode == 'train':
                os.makedirs(self.path_model)
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



