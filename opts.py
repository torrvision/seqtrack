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
        self.verbose_train      = False
        self.report             = False
        self.debugmode          = False # True or False
        self.histograms         = False # True or False
        self.tfdb               = False # True or False
        self.seed_global        = 9
        self.dtype              = tf.float32
        self.exectime           = helpers.get_time()
        self.tfversion          = tf.__version__[0:4]

        #----------------------------------------------------------------------
        # data set specific parameters 
        # TODO: only params that need to change; otherwise put it in data class
        self.dataset            = '' # (bouncing_mnist, etc.)
        self.trainsplit         = 9 # 0,1,2,3 or 9 for all train sets
        self.frmsz              = 241
        self.useresizedimg      = True
        self.use_queues         = False

        #----------------------------------------------------------------------
        # model - general
        self.model              = '' # {rnn_basic, rnn_attention_s, rnn_attention_t}
        self.losses             = None

        #----------------------------------------------------------------------
        # model parameters - rnn
        self.nunits             = 256
        self.ntimesteps         = 20

        #----------------------------------------------------------------------
        # model parameters - cnn (or feature extractor)
        self.cnn_pretrain       = False 
        self.cnn_model          = 'vgg' # vgg, resnet, imagenet, etc.
        self.model_params       = {}

        #----------------------------------------------------------------------
        # training policies
        self.nepoch             = 20
        self.batchsz            = 1
        self.optimizer          = 'adam' # sgd, adam, rmsprop
        self.lr_init            = 1e-4
        self.lr_decay_rate      = 1 # No decay.
        self.lr_decay_steps     = 10000
        self.wd                 = 0.0 # weight decay for regularization
        self.grad_clip          = False
        self.max_grad_norm      = 5.0
        self.gt_decay_rate      = -1e-2
        self.min_gt_ratio       = 0.75
        self.curriculum_learning= False
        self.model_file         = None

        #----------------------------------------------------------------------
        # Options for sampler
        # The sampler to use for training.
        # Do not specify `ntimesteps` or `shuffle` here.
        self.sampler_params     = {'kind': 'regular', 'freq': 10}
        # Dataset and sampler to use for evaluation.
        self.eval_datasets      = ['ILSVRC-train']
        self.eval_samplers      = ['train']
        self.max_eval_videos    = 100

        #----------------------------------------------------------------------
        # save (save training results), load (test), resume (keep training)
        self.path_src           = os.path.dirname(__file__) # Take aux/ and stat/ from source dir.
        self.path_aux           = os.path.join(self.path_src, 'aux')
        self.path_stat          = os.path.join(self.path_src, 'stat')
        self.path_data_home     = './data'
        self.path_data          = '' # This is set later e.g. {path_data_home}/ILSVRC
        self.nosave             = False
        self.path_ckpt          = './ckpt'
        self.path_output        = './output'
        self.path_summary       = './summary'
        self.resume             = False
        self.period_ckpt        = 10000 # this is based only on global_step; batchsz not considered
        self.period_assess      = 10000
        self.period_summary     = 10
        self.period_preview     = 100 # Ensure that period_preview % period_summary == 0.
        self.visualize_eval     = False

        #----------------------------------------------------------------------
        # custom libraries
        self.path_home          = os.path.expanduser('~')
        self.path_customlib     = os.path.join(self.path_home, 
                                'tensorflow/bazel-bin/tensorflow/core/user_ops') 

        #----------------------------------------------------------------------
        # device memory allocation 
        self.gpu_device         = 0 # set `CUDA_VISIBLE_DEVICES` gpu number
        self.gpu_manctrl        = True # set it always True and control gpu_frac
        self.gpu_frac           = 0.4 # option for memory allocation

    
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
        self.activ_histogram = self.histograms
        self.param_histogram = self.histograms

    def _set_dataset_params(self):
        assert(self.dataset in ['moving_mnist', 'bouncing_mnist', 'ILSVRC'])
        self.path_data = os.path.join(self.path_data_home, self.dataset)

    def _set_gpu_config(self):
        # set `CUDA_VISIBLE_DEVICES`
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(self.gpu_device)
        # set tfconfig 
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
        pass


if __name__ == '__main__':
    '''Test options
    '''
    o = Opts()
    pdb.set_trace()
