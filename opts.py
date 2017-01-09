import pdb
import numpy as np
import os

class Opts(object):
    def __init__(self):
        ''' Initial parameter settings 
        '''
        #----------------------------------------------------------------------
        # basic settings 
        self.path_base = os.path.dirname(__file__)
        #self.path_data = self.path_base + '/data' 
        
        #----------------------------------------------------------------------
        # model parameters

        #----------------------------------------------------------------------
        # training policies

        # test params
        self.testval = 10
    
    def update_sysarg(self, args):
        ''' Update options by system arguments
        '''
        for ko, vo in vars(self).items():
            for ka, va in vars(args).items():
                if ko == ka and vo != va:
                    setattr(self, ko, va)
                    print 'opt changed ({}): {} -> {}'.format(ko, vo, va)



