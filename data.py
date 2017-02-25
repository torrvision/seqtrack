import pdb
import numpy as np
import cPickle as pickle
import gzip

# packages for bouncing mnist
import h5py
import scipy.ndimage as spn
import os

import draw

import glob
import random
import xmltodict
import cv2

import helpers


class Data_moving_mnist(object):
    '''
    The main routine to generate moving mnist is derived from 
    "RATM: Recurrent Attentive Tracking Model"
    '''
    def __init__(self, o):
        #self.datafile = o.path_data+'/'+o.dataset+'/mnist.pkl.gz' 
        self.datafile = o.path_data + '/mnist.pkl.gz'
        self.frmsz = o.frmsz
        self.ninchannel = o.ninchannel
        self.outdim = o.outdim

        self.data = dict.fromkeys({'train', 'val', 'test'}, None)
        self.nexps = dict.fromkeys({'train', 'val', 'test'}, None)
        self.idx_shuffle = dict.fromkeys({'train', 'val', 'test'}, None)

        self._load_data()
        self._update_idx_shuffle(['train', 'val', 'test'])

    def _load_data(self):
        with gzip.open(self.datafile, 'rb') as f:
             rawdata = pickle.load(f)
        for p, part in enumerate(('train', 'val', 'test')):
            self.data[part] = {}
            self.data[part]['images'] = rawdata[p][0].reshape(-1, 28, 28)
            self.data[part]['targets'] = rawdata[p][1]
            self.nexps[part] = self.data[part]['images'].shape[0]

    def _update_idx_shuffle(self, dstypes):
        for dstype in dstypes:
            if dstype in self.nexps and self.nexps[dstype] is not None:
                self.idx_shuffle[dstype] = np.random.permutation(self.nexps[dstype])

    def get_batch(self, ib, o, dstype, shuffle_local=False):
        '''
        Everytime this function is called, create the batch number of moving 
        mnist sets. If this process has randomness, no other data augmentation 
        technique is applied for now. 
        '''
        data = self.data[dstype]
        if shuffle_local: # used for evaluation during train
            idx = np.random.permutation(self.nexps[dstype])[(ib*o.batchsz):(ib+1)*o.batchsz]
        else:
            idx = self.idx_shuffle[dstype][(ib*o.batchsz):(ib+1)*o.batchsz] 

        # the following is a modified version from RATM data preparation
        vids = np.zeros((o.batchsz, o.ntimesteps+1, self.frmsz, self.frmsz), 
                dtype=np.float32)
        pos_init = np.random.randint(self.frmsz-28, size=(o.batchsz,2))
        pos = np.zeros((o.batchsz, o.ntimesteps+1, 2), dtype=np.int32)
        pos[:,0] = pos_init

        posmax = self.frmsz-29

        d = np.random.randint(low=-15, high=15, size=(o.batchsz,2))

        for t in range(o.ntimesteps+1):
            dtm1 = d
            d = np.random.randint(low=-15, high=15, size=(o.batchsz,2))
            for i in range(o.batchsz):
                '''
                vids[i,t,
                        pos[i,t,0]:pos[i,t,0]+28,
                        pos[i,t,1]:pos[i,t,1]+28] = \
                                data['images'][idx[i]]
                '''
                # This makes pos -> (x,y) instead of (y,x)
                vids[i,t,
                        pos[i,t,1]:pos[i,t,1]+28,
                        pos[i,t,0]:pos[i,t,0]+28] = \
                                data['images'][idx[i]]
            if t < o.ntimesteps+1-1:
                pos[:,t+1] = pos[:,t]+.1*d+.9*dtm1

                # check for proposer position (reflect if necessary)
                reflectidx = np.where(pos[:,t+1] > posmax)
                pos[:,t+1][reflectidx] = (posmax - 
                        (pos[:,t+1][reflectidx] % posmax))
                reflectidx = np.where(pos[:,t+1] < 0)
                pos[:,t+1][reflectidx] = -pos[:,t+1][reflectidx]

        # TODO: variable length inputs for
        # 1. online learning, 2. arbitrary training sequences 
        inputs_length = np.ones((o.batchsz), dtype=np.int32) * (o.ntimesteps+1)

        inputs_HW = np.ones((o.batchsz, 2), dtype=np.float32) * self.frmsz

        # Add fixed sized window to make label 4 dimension. [only moving_mnist]
        pos = np.concatenate((pos, pos+28), axis=2)

        # relative scale of label
        pos = pos / float(o.frmsz)

        # add one more dimension to data for placeholder tensor shape
        vids = np.expand_dims(vids, axis=4)

        batch = {
                'inputs': vids,
                'inputs_length': inputs_length, 
                'inputs_HW': inputs_HW, 
                'labels': pos,
                'digits': data['targets'][idx],
                'idx': idx
                }
        return batch

    def update_epoch_begin(self, dstype):
        # may no need dstype, as this shuffles train data only.
        self.idx_shuffle[dstype] = np.random.permutation(self.nexps[dstype])

    def run_sanitycheck(self, batch, dataset, frmsz):
        draw.show_dataset_batch(batch, dataset, frmsz)


class Data_bouncing_mnist(object):
    '''
    The main routine to generate bouncing mnist is derived from 
    "First Step toward Model-Free, Anonymous Object Tracking with 
    Recurrent Neural Networks"
    
    The original code is written quite poorly, and is not suitable for direct
    use (e.g., the data loading and splits are mixed, only batch loading, etc.)
    so I made changes in quite many places.
    '''

    def __init__(self, o):
        #self.data_path = os.path.join(o.path_base, 'data/bouncing_mnist')
        self.data_path = o.path_data
        self._set_default_params(o)

        self.data = dict.fromkeys({'train', 'val', 'test'}, None)
        self.nexps = dict.fromkeys({'train', 'val', 'test'}, None)
        self.idx_shuffle = dict.fromkeys({'train', 'val', 'test'}, None)

        #self.data, self.label = None, None
        self._load_data()
        self._update_idx_shuffle(['train', 'val', 'test'])

    def _set_default_params(self, o):
        # following paper's default parameter settings
        # TODO: remember some variables might need to set optional..
        self.num_digits_ = 1 
        self.frmsz = o.frmsz
        self.scale_range = 0.1 
        self.buff = True
        self.step_length_ = 0.1
        self.digit_size_ = 28
        self.frame_size_ = self.frmsz ** 2
        self.dataset_size_ = 10000  # Size is relevant only for val/test sets.
        #self.row_ = 0 # NL: should not use it!!!
        self.clutter_size_min_ = 5
        self.clutter_size_max_ = 10
        self.num_clutters_ = 20
        self.face_intensity_min = 64
        self.face_intensity_max = 255
        self.acc_scale = 0.1 
        self.vel_scale = 1
        #self.indices_ = np.arange(self.data.shape[0]) # changed to idx_shuffle
        #np.random.shuffle(self.indices_) 
        self.num_clutterPack = 10000
        self.clutterpack_exists = os.path.exists(os.path.join(
            self.data_path, 'ClutterPackLarge.hdf5'))
        if not self.clutterpack_exists:
            self._InitClutterPack()
        f = h5py.File(os.path.join(self.data_path,'ClutterPackLarge.hdf5'), 'r')
        self.clutterPack = f['clutterIMG'][:]
        self.buff_ptr = 0
        self.buff_size = 2000
        self.buff_cap = 0
        self.buff_data = np.zeros((self.buff_size, o.ntimesteps+1, 
            self.frmsz, self.frmsz), dtype=np.float32)
        self.buff_label = np.zeros((self.buff_size, o.ntimesteps+1, 4))
        self.clutter_move = 1 
        self.with_clutters = 1 

    def _load_data(self):
        f = h5py.File(os.path.join(self.data_path, 'mnist.h5'))
        # f has 'train' and 'test' and each has 'inputs' and 'targets'
        for dstype in ['train', 'test']:
            self.data[dstype] = dict.fromkeys({'images', 'targets'}, None) 
            self.data[dstype]['images'] = np.asarray(
                    f['{}/inputs'.format(dstype)].value)
            self.data[dstype]['targets'] = np.asarray(
                    f['{}/targets'.format(dstype)].value)
            self.nexps[dstype] = self.data[dstype]['images'].shape[0]
        f.close()

        # TODO: from the original code, they further separate train/test by
        # actual digits, as the following: 
        '''
        if run_flag == 'train':
            idx = np.where(self.label<5)[0]
            self.data = self.data[idx]
        if run_flag == 'test':
            idx = np.where(self.label>4)[0]
            self.data = self.data[idx]
        '''

    def _GetRandomTrajectory(
            self, batch_size, o,
            image_size_=None, object_size_=None, step_length_=None):
        if image_size_ is None:
            image_size_ = self.frmsz
        if object_size_ is None:
            object_size_ = self.digit_size_
        if step_length_ is None:
            step_length_ = self.step_length_
        length = o.ntimesteps+1
        canvas_size = image_size_ - object_size_

        # Initial position uniform random inside the box.
        y = np.random.rand(batch_size)
        x = np.random.rand(batch_size)

        # Choose a random velocity.
        theta = np.random.rand(batch_size) * 2 * np.pi
        start_vel = np.random.normal(0, self.vel_scale)
        v_y = start_vel * np.sin(theta)
        v_x = start_vel * np.cos(theta)

        start_y = np.zeros((length, batch_size))
        start_x = np.zeros((length, batch_size))
        for i in range(length):
            # Take a step along velocity.
            y += v_y * step_length_
            x += v_x * step_length_

            v_y += 0 if self.acc_scale == 0 else np.random.normal(0, self.acc_scale, v_y.shape)
            v_x += 0 if self.acc_scale == 0 else np.random.normal(0, self.acc_scale, v_x.shape)

            # Bounce off edges.
            for j in range(batch_size):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
                start_y[i, :] = y
                start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def _Overlap(self, a, b):
        """ Put b on top of a."""
        b = np.where(b > (np.max(b) / 4), b, 0)
        t = min(np.shape(a))
        b = b[:t, :t]
        return np.select([b == 0, b != 0], [a, b])
        #return b

    def _InitClutterPack(self, num_clutterPack = None, image_size_ = None, num_clutters_ = None):
        if num_clutterPack is None :
            num_clutterPack = self.num_clutterPack
        if image_size_ is None :
            image_size_ = self.frmsz * 2
        if num_clutters_ is None :
            num_clutters_ = self.num_clutters_ * 4
        clutterIMG = np.zeros((num_clutterPack, image_size_, image_size_))
        for i in xrange(num_clutterPack):
            #clutterIMG[i] = self._GetClutter(image_size_, num_clutters_)
            clutterIMG[i] = self._GetClutter(image_size_, num_clutters_, 
                    dstype='train') # TODO: this should be fine
        f = h5py.File(os.path.join(self.data_path,'ClutterPackLarge.hdf5', 'w'))
        f.create_dataset('clutterIMG', data=clutterIMG)
        f.close()
            
    def _GetFakeClutter(self):
        if self.clutterpack_exists:
            return self.clutterPack[np.random.randint(0, len(self.clutterPack))]
    
    def _GetClutter(self, image_size_ = None, num_clutters_ = None, fake = False, dstype=None):
        data_all = self.data[dstype]

        if image_size_ is None :
            image_size_ = self.frmsz
        if num_clutters_ is None :
            num_clutters_ = self.num_clutters_
        if fake and self.clutterpack_exists:
            return self._GetFakeClutter()
        clutter = np.zeros((image_size_, image_size_), dtype=np.float32)
        for i in range(num_clutters_):
            sample_index = np.random.randint(data_all['images'].shape[0])
            size = np.random.randint(self.clutter_size_min_, self.clutter_size_max_)
            left = np.random.randint(0, self.digit_size_ - size)
            top = np.random.randint(0, self.digit_size_ - size)
            clutter_left = np.random.randint(0, image_size_ - size)
            clutter_top = np.random.randint(0, image_size_ - size)
            single_clutter = np.zeros_like(clutter)
            single_clutter[clutter_top:clutter_top+size, clutter_left:clutter_left+size] = data_all['images'][np.random.randint(data_all['images'].shape[0]), top:top+size, left:left+size] / 255.0 * np.random.uniform(self.face_intensity_min, self.face_intensity_max)
            clutter = self._Overlap(clutter, single_clutter)
        return clutter

    def _getBuff(self):
        #print 'getBuff ',
        idx = np.random.randint(0, self.buff_cap)
        return self.buff_data[idx], self.buff_label[idx]

    def _setBuff(self, data, label):
        self.buff_data[self.buff_ptr]=data
        self.buff_label[self.buff_ptr]=label
        if self.buff_cap < self.buff_size:
            self.buff_cap += 1
        self.buff_ptr += 1
        self.buff_ptr = self.buff_ptr % self.buff_size

    def _update_idx_shuffle(self, dstypes):
        for dstype in dstypes:
            if dstype in self.nexps and self.nexps[dstype] is not None:
                self.idx_shuffle[dstype] = np.random.permutation(self.nexps[dstype])

    def get_batch(self, ib, o, dstype, verbose=False, count=1, shuffle_local=False):
        '''Here in this function also made several changes in several places.
        '''
        data_all = self.data[dstype]
        if shuffle_local: # used for evaluation during train
            idx = np.random.permutation(self.nexps[dstype])[(ib*o.batchsz):(ib+1)*o.batchsz]
        else:
            idx = self.idx_shuffle[dstype][(ib*o.batchsz):(ib+1)*o.batchsz] 
        
        data = np.zeros((o.batchsz, o.ntimesteps+1, self.frmsz, self.frmsz), dtype=np.float32)
        label = np.zeros((o.batchsz, o.ntimesteps+1, 4), dtype=np.float32)

        start_y, start_x = self._GetRandomTrajectory(o.batchsz * self.num_digits_, o)
        window_y, window_x = self._GetRandomTrajectory(o.batchsz * 1, o, self.frmsz*2, object_size_=self.frmsz, step_length_ = 1e-2)
        # TODO: change data to real image or cluttered background

        for j in range(o.batchsz): 
            if np.random.random()<0.7 and self.buff and self.buff_cap > self.buff_size/2.0:
                data[j], label[j] = self._getBuff()
                continue
            else:
                clutter = self._GetClutter(fake=True, dstype=dstype)
                clutter_bg = self._GetClutter(fake=True, dstype=dstype)
                wc = np.random.ranf() < self.with_clutters
                cm = np.random.ranf() < self.clutter_move
                if wc:
                    if cm:
                        for i in range(o.ntimesteps+1):
                            wx = window_x[i,j]
                            wy = window_y[i,j]
                            data[j, i] = self._Overlap(clutter_bg[wy:wy+self.frmsz, wx:wx+self.frmsz], data[j, i])
                    else:
                        for i in range(o.ntimesteps+1):
                            wx = window_x[0, j]
                            wy = window_y[0, j]
                            data[j, i] = self._Overlap(clutter_bg[wy:wy+self.frmsz, wx:wx+self.frmsz], data[j, i])
                for n in range(self.num_digits_):
                    #ind = self.indices_[self.row_]
                    ind = idx[j]
                    ''' NL: no need this 
                    self.row_ += 1
                    if self.row_ == data_all['images'].shape[0]:
                        self.row_ = 0
                        #np.random.shuffle(self.indices_)
                        np.random.shuffle(idx_shuffle)
                    '''
                    if count == 2:
                        digit_image = np.zeros((data_all['images'].shape[1], data_all['images'].shape[2]))
                        digit_image[:18, :18] = self._Overlap(digit_image[:18, :18], np.maximum.reduceat(np.maximum.reduceat(data_all['images'][ind], np.cast[int](np.arange(1, 28, 1.5))), np.cast[int](np.arange(1, 28, 1.5)), axis=1))
                        digit_image[10:, 10:] = self._Overlap(digit_image[10:, 10:], np.maximum.reduceat(np.maximum.reduceat(data_all['images'][np.random.randint(data_all['images'].shape[0])], np.cast[int](np.arange(0, 27, 1.5))), np.cast[int](np.arange(0, 27, 1.5)), axis=1))
                    else:
                        digit_image = data_all['images'][ind, :, :] / 255.0 * np.random.uniform(self.face_intensity_min, self.face_intensity_max)
                    bak_digit_image = digit_image 
                    digit_size_ = self.digit_size_
                    for i in range(o.ntimesteps+1):
                        scale_factor = np.exp((np.random.random_sample()-0.5)*self.scale_range)
                        scale_image = spn.zoom(digit_image, scale_factor)
                        digit_size_ = digit_size_ * scale_factor 
                        top    = start_y[i, j * self.num_digits_ + n]
                        left   = start_x[i, j * self.num_digits_ + n]
                        if digit_size_!=np.shape(scale_image)[0]:
                            digit_size_ = np.shape(scale_image)[0]
                        bottom = top  + digit_size_
                        right  = left + digit_size_
                        if right>self.frmsz or bottom>self.frmsz:
                            scale_image = bak_digit_image
                            bottom = top  + self.digit_size_
                            right  = left + self.digit_size_
                            digit_size_ = self.digit_size_
                        digit_image = scale_image
                        digit_image_nonzero = np.where(digit_image > (np.max(digit_image) / 4), digit_image, 0).nonzero()
                        # NL: (y,x) -> (x,y)
                        #label_offset = np.array([digit_image_nonzero[0].min(), digit_image_nonzero[1].min(), digit_image_nonzero[0].max(), digit_image_nonzero[1].max()])
                        label_offset = np.array([digit_image_nonzero[1].min(), digit_image_nonzero[0].min(), digit_image_nonzero[1].max(), digit_image_nonzero[0].max()])
 
                        wy=window_y[i, j]
                        wx=window_x[i, j]
                        data[j, i, top:bottom, left:right] = self._Overlap(data[j, i, top:bottom, left:right], scale_image)
                        data[j, i] = self._Overlap(data[j, i], clutter[wy:wy+self.frmsz, wx:wx+self.frmsz])
                        # NL: (y,x) -> (x,y)
                        #label[j, i] = label_offset + np.array([top, left, top, left])
                        label[j, i] = label_offset + np.array([left, top, left, top])
                if wc:
                    if cm:
                        for i in range(o.ntimesteps+1):
                            wx = window_x[i,j]
                            wy = window_y[i,j]
                            data[j, i] = self._Overlap(data[j, i], clutter[wy:wy+self.frmsz, wx:wx+self.frmsz])
                    else:
                        for i in range(o.ntimesteps+1):
                            wx = window_x[0,j]
                            wy = window_y[0,j]
                            data[j, i] = self._Overlap(data[j, i], clutter[wy:wy+self.frmsz, wx:wx+self.frmsz])
                if self.buff:
                    self._setBuff(data[j], label[j])

        # TODO: variable length inputs for
        # 1. online learning, 2. arbitrary training sequences 
        inputs_length = np.ones((o.batchsz), dtype=np.int32) * (o.ntimesteps+1)

        inputs_HW = np.ones((o.batchsz, 2), dtype=np.float32) * self.frmsz

        # relative scale of label
        label = label / o.frmsz

        # add one more dimension to data for placeholder tensor shape
        data = np.expand_dims(data, axis=4)

        batch = {
                'inputs': data,
                'inputs_length': inputs_length, 
                'inputs_HW': inputs_HW,
                'labels': label,
                'digits': data_all['targets'][idx],
                'idx': idx
                }

        return batch

    def get_image(self, idx, dstype):
        #NOTE: This method can't give you all images in a sequence! (not useful)
        return self.data[dstype]['images'][idx]

    def update_epoch_begin(self, dstype):
        # may no need dstype, as this shuffles train data only.
        self.idx_shuffle[dstype] = np.random.permutation(self.nexps[dstype])

    def run_sanitycheck(self, batch, dataset, frmsz):
        draw.show_dataset_batch(batch, dataset, frmsz)


class Data_ILSVRC(object):
    def __init__(self, o):
        self.path_data      = o.path_data
        self.trainsplit     = o.trainsplit # 0, 1, 2, 3, or 9 for all
        self.datadirname    = 'Data_frmsz{}'.format(o.frmsz) \
                                if o.useresizedimg else 'Data'

        self.snps               = dict.fromkeys({'train', 'val', 'test'}, None)
        self.nsnps              = dict.fromkeys({'train', 'val', 'test'}, None)
        self.nfrms_snp          = dict.fromkeys({'train', 'val', 'test'}, None)
        self.objids_allfrm_snp  = dict.fromkeys({'train', 'val', 'test'}, None)
        self.objids_snp         = dict.fromkeys({'train', 'val', 'test'}, None)
        self.objids_valid_snp   = dict.fromkeys({'train', 'val', 'test'}, None)
        self.objvalidfrms_snp   = dict.fromkeys({'train', 'val', 'test'}, None)

        self.nexps              = dict.fromkeys({'train', 'val', 'test'}, None)

        self.idx_shuffle        = dict.fromkeys({'train', 'val', 'test'}, None)
        self.stat               = dict.fromkeys({'train', 'val', 'test'}, None)

        self._load_data(o)
        self._update_idx_shuffle(['train', 'val', 'test'])

    def _load_data(self, o):
        # TODO: need to process and load test data set as well..
        # TODO: not loading test data yet. ILSVRC doesn't have "Annotations" for
        # test set. I can still load "Data", but need to fix a couple of places
        # to load only "Data". Will fix it later.
        for dstype in ['train', 'val']: # ['train', 'val', 'test']:
            self._update_snps(dstype)
            self._update_nsnps(dstype)
            self._update_nfrms_snp(dstype)
            self._update_objids_allfrm_snp(dstype, o)
            self._update_objids_snp(dstype) # simply unique obj ids in a snippet
            self._update_objvalidfrms_snp(dstype)
            self._update_nexps(dstype)
            self._update_stat(dstype, o)

    def _parsexml(self, xmlfile):
        with open(xmlfile) as f:
            doc = xmltodict.parse(f.read())
        return doc

    def _update_snps(self, dstype):
        def create_snps(dstype):
            output = {}
            if dstype is 'test': # Only data exists. No annotations available. 
                path_snp_data = os.path.join(
                    self.path_data, '{}/VID/{}'.format(self.datadirname, dstype))
                snps = os.listdir(path_snp_data)
                snps_data = [path_snp_data + '/' + snp for snp in snps]
                output['Data'] = snps_data
            elif dstype is 'val': 
                path_snp_data = os.path.join(
                    self.path_data, '{}/VID/{}'.format(self.datadirname, dstype))
                path_snp_anno = os.path.join(
                    self.path_data, 'Annotations/VID/{}'.format(dstype))
                snps = os.listdir(path_snp_data)
                snps_data = [path_snp_data + '/' + snp for snp in snps]
                snps_anno = [path_snp_anno + '/' + snp for snp in snps]
                output['Data'] = snps_data
                output['Annotations'] = snps_anno
            elif dstype is 'train': # train data has 4 splits.
                if self.trainsplit in [0,1,2,3]:
                    path_snp_data = os.path.join(self.path_data, 
                        '{0:s}/VID/train/ILSVRC2015_VID_train_{1:04d}'.format(self.datadirname, self.trainsplit))
                    path_snp_anno = os.path.join(self.path_data, 
                        'Annotations/VID/train/ILSVRC2015_VID_train_{0:04d}'.format(self.trainsplit))
                    snps = os.listdir(path_snp_data)
                    snps_data = [path_snp_data + '/' + snp for snp in snps]
                    snps_anno = [path_snp_anno + '/' + snp for snp in snps]
                    output['Data'] = snps_data
                    output['Annotations'] = snps_anno
                elif self.trainsplit == 9: # MAGIC NUMBER for all train data 
                    snps_data_all = []
                    snps_anno_all = []
                    for i in range(4):
                        path_snp_data = os.path.join(self.path_data, 
                            '{0:s}/VID/train/ILSVRC2015_VID_train_{1:04d}'.format(self.datadirname, i))
                        path_snp_anno = os.path.join(self.path_data, 
                            'Annotations/VID/train/ILSVRC2015_VID_train_{0:04d}'.format(i))
                        snps = os.listdir(path_snp_data)
                        snps_data = [path_snp_data + '/' + snp for snp in snps]
                        snps_anno = [path_snp_anno + '/' + snp for snp in snps]
                        snps_data_all.extend(snps_data)
                        snps_anno_all.extend(snps_anno)
                    output['Data'] = snps_data_all
                    output['Annotations'] = snps_anno_all
                else:
                    raise ValueError('No available option for train split')
            return output
        if self.snps[dstype] is None:
            self.snps[dstype] = create_snps(dstype)

    def _update_nsnps(self, dstype):
        if self.nsnps[dstype] is None:
            self.nsnps[dstype] = len(self.snps[dstype]['Data'])

    def _update_nfrms_snp(self, dstype):
        def create_nfrms_snp(dstype):
            nfrms = []
            for snp in self.snps[dstype]['Data']:
                nfrms.append(len(glob.glob(snp+'/*.JPEG')))
            return nfrms
        if self.nfrms_snp[dstype] is None:
            self.nfrms_snp[dstype] = create_nfrms_snp(dstype)

    def _update_objids_allfrm_snp(self, dstype, o):
        def extract_objids_allfrm_snp(dstype):
            def extract_objids_from_xml(xmlfile, doc=None):
                if doc is None: 
                    doc = self._parsexml(xmlfile)
                if 'object' in doc['annotation']:
                    if type(doc['annotation']['object']) is list:
                        trackids = []
                        for i in range(len(doc['annotation']['object'])):
                            trackids.append(
                                int(doc['annotation']['object'][i]['trackid']))
                        return trackids 
                    else:
                        return [int(doc['annotation']['object']['trackid'])]
                else:
                    return [None] # No object in this file (or current frame)

            objids_allfrm_snp = []
            for i in range(self.nsnps[dstype]):
                print i
                objids_frm = []
                for j in range(self.nfrms_snp[dstype][i]):
                    xmlfile = os.path.join(self.snps[dstype]['Annotations'][i], 
                        '{0:06d}.xml'.format(j))
                    objids_frm.append(extract_objids_from_xml(xmlfile))
                objids_allfrm_snp.append(objids_frm)
            return objids_allfrm_snp
    
        if self.objids_allfrm_snp[dstype] is None:
            if dstype == 'train':
                filename = os.path.join(o.path_aux, 
                    'objids_allfrm_snp_{}_{}.npy'.format(dstype, self.trainsplit))
            else:
                filename = os.path.join(o.path_aux, 
                    'objids_allfrm_snp_{}.npy'.format(dstype))
            if os.path.exists(filename):
                self.objids_allfrm_snp[dstype] = np.load(filename).tolist()
            else: # if no file, create and also save
                self.objids_allfrm_snp[dstype] = extract_objids_allfrm_snp(dstype)
                np.save(filename, self.objids_allfrm_snp[dstype])

    def _update_objids_snp(self, dstype):
        assert(self.objids_allfrm_snp[dstype] is not None) # obtain from 'objids_allfrm_snp'
        def find_objids_snp(dstype):
            output = []
            for objids_allfrm_snp in self.objids_allfrm_snp[dstype]:
                objids_unique = set([item for sublist in objids_allfrm_snp for item in sublist])
                objids_unique.discard(None)
                assert(len(objids_unique)>0)
                output.append(objids_unique)
            return output
        if self.objids_snp[dstype] is None:
            self.objids_snp[dstype] = find_objids_snp(dstype)

    def _update_objvalidfrms_snp(self, dstype):
        assert(self.objids_allfrm_snp[dstype] is not None) # obtain from 'objids_allfrm_snp'
        def extract_objvalidfrms_snp(dstype):
            objvalidfrms_snp = []
            # NOTE: there are objs that only appear one frame. 
            # This objids are not valid. Thus, objids_valid_snp comes in.
            objids_valid_snp = []
            for i in range(self.nsnps[dstype]): # snippets
                #print 'processing objvalidfrms_snp {}'.format(i)
                objvalidfrms = {}
                objids_valid = []
                for objid in self.objids_snp[dstype][i]: # objects
                    validfrms = []
                    flag_one = False
                    flag_consecutive_one = False
                    for t in range(self.nfrms_snp[dstype][i]):
                        if objid in self.objids_allfrm_snp[dstype][i][t]:
                            validfrms.append(1)
                            if flag_one:
                                flag_consecutive_one = True
                            flag_one = True
                        else:
                            validfrms.append(0)
                            flag_one = False
                    assert(np.sum(validfrms)> 0)
                    if flag_consecutive_one: # the object should be seen at least 2 consecutive frames
                        objvalidfrms[objid] = validfrms
                        objids_valid.append(objid)

                # check if there is not a single available objvalidfrms
                assert(len(objvalidfrms)>0)
                objvalidfrms_snp.append(objvalidfrms)
                objids_valid_snp.append(set(objids_valid))
            self.objids_valid_snp[dstype] = objids_valid_snp
            return objvalidfrms_snp

        if self.objvalidfrms_snp[dstype] is None:
            self.objvalidfrms_snp[dstype] = extract_objvalidfrms_snp(dstype)

    def _update_nexps(self, dstype):
        if self.nexps[dstype] is None:
            self.nexps[dstype] = self.nsnps[dstype]

    def _update_idx_shuffle(self, dstypes):
        for dstype in dstypes:
            if dstype in self.nexps and self.nexps[dstype] is not None:
                self.idx_shuffle[dstype] = np.random.permutation(self.nexps[dstype])

    def _update_stat(self, dstype, o):
        def create_stat_pixelwise(dstype): # NOTE: obsolete
            stat = dict.fromkeys({'mean', 'std'}, None)
            mean = []
            std = []
            for i, snp in enumerate(self.snps[dstype]['Data']):
                print 'computing mean and std in snippet of {}, {}/{}'.format(
                        dstype, i+1, self.nsnps[dstype])
                imglist = glob.glob(snp+'/*.JPEG')
                xs = []
                for j in imglist:
                    # NOTE: perform resize image!
                    xs.append(cv2.resize(cv2.imread(j)[:,:,(2,1,0)], 
                        (o.frmsz, o.frmsz), interpolation=cv2.INTER_AREA))
                xs = np.asarray(xs)
                mean.append(np.mean(xs, axis=0))
                std.append(np.std(xs, axis=0))
            mean = np.mean(np.asarray(mean), axis=0)
            std = np.mean(np.asarray(std), axis=0)
            stat['mean'] = mean
            stat['std'] = std
            return stat
        def create_stat_global(dstype):
            stat = dict.fromkeys({'mean', 'std'}, None)
            means = []
            stds = []
            for i, snp in enumerate(self.snps[dstype]['Data']):
                print 'computing mean and std in snippet of {}, {}/{}'.format(
                        dstype, i+1, self.nsnps[dstype])
                imglist = glob.glob(snp+'/*.JPEG')
                xs = []
                for j in imglist:
                    # NOTE: perform resize image!
                    xs.append(cv2.resize(cv2.imread(j)[:,:,(2,1,0)], 
                        (o.frmsz, o.frmsz), interpolation=cv2.INTER_AREA))
                means.append(np.mean(xs))
                stds.append(np.std(xs))
            mean = np.mean(means)
            std = np.mean(stds)
            stat['mean'] = mean
            stat['std'] = std
            return stat

        if self.stat[dstype] is None:
            if dstype == 'train':
                filename = os.path.join(o.path_stat, 
                    'meanstd_{}_frmsz_{}_train_{}.npy'.format(o.dataset, o.frmsz, self.trainsplit))
            else:
                filename = os.path.join(o.path_stat,
                    'meanstd_{}_frmsz_{}_{}.npy'.format(o.dataset, o.frmsz, dstype))
            if os.path.exists(filename):
                self.stat[dstype] = np.load(filename).tolist()
            else:
                self.stat[dstype] = create_stat_global(dstype) 
                np.save(filename, self.stat[dstype])

    def get_batch(self, ib, o, dstype, shuffle_local=False):
        def select_frms(objvalidfrms):
            # firstly create consecutive 1s 
            segment_minlen = 2
            consecutiveones = []
            stack = []
            for i, val in enumerate(objvalidfrms):
                if val == 0: 
                    if len(stack) >= segment_minlen:
                        consecutiveones.append(stack)
                    stack = []
                elif val == 1:
                    stack.append(i)
                else:
                    raise ValueError('should be either 1 or 0')
            if len(stack) >= segment_minlen: consecutiveones.append(stack)

            # randomly choose one segment
            frms_cand = random.choice(consecutiveones)

            # select frames (randomness in it and < RNN+1 size)
            frm_length = np.minimum(
                random.randint(segment_minlen, len(frms_cand)), o.ntimesteps+1)
            frm_start = random.randint(0, len(frms_cand)-frm_length)
            frms = frms_cand[frm_start:frm_start+frm_length]
            return frms

        def get_bndbox_from_xml(xmlfile, objid):
            doc = self._parsexml(xmlfile)
            w = np.float32(doc['annotation']['size']['width'])
            h = np.float32(doc['annotation']['size']['height'])
            # NOTE: Case of no object in the current frame.
            # Either None or zeros. None becomes 'nan' when converting to numpy.
            #bndbox = [None, None, None, None]
            bndbox = [0, 0, 0, 0]
            if 'object' in doc['annotation']:
                if type(doc['annotation']['object']) is list:
                    nobjs = len(doc['annotation']['object'])
                    for i in range(nobjs):
                        if int(doc['annotation']['object'][i]['trackid']) == objid:
                            bndbox = [
                                np.float32(doc['annotation']['object'][i]['bndbox']['xmin']) / w,
                                np.float32(doc['annotation']['object'][i]['bndbox']['ymin']) / h,
                                np.float32(doc['annotation']['object'][i]['bndbox']['xmax']) / w,
                                np.float32(doc['annotation']['object'][i]['bndbox']['ymax']) / h]
                            break
                else:
                    if int(doc['annotation']['object']['trackid']) == objid:
                        bndbox = [
                            np.float32(doc['annotation']['object']['bndbox']['xmin']) / w,
                            np.float32(doc['annotation']['object']['bndbox']['ymin']) / h,
                            np.float32(doc['annotation']['object']['bndbox']['xmax']) / w,
                            np.float32(doc['annotation']['object']['bndbox']['ymax']) / h]
            else:
                # NOTE: this case should be considered later.
                raise ValueError('currently not allowing no labeled frames')
            return bndbox # xyxy format

        if shuffle_local: # used for evaluation during train
            idx = np.random.permutation(self.nexps[dstype])[(ib*o.batchsz):(ib+1)*o.batchsz]
        else:
            idx = self.idx_shuffle[dstype][(ib*o.batchsz):(ib+1)*o.batchsz] 

        # NOTE: examples have a length of ntimesteps+1.
        data = np.zeros(
            (o.batchsz, o.ntimesteps+1, o.frmsz, o.frmsz, o.ninchannel), 
            dtype=np.float32)
        label = np.zeros((o.batchsz, o.ntimesteps+1, o.outdim), dtype=np.float32)
        #inputs_valid = np.zeros((o.batchsz, o.ntimesteps+1), dtype=np.int32)
        inputs_length = np.zeros((o.batchsz), dtype=np.int32)
        inputs_HW = np.zeros((o.batchsz, 2), dtype=np.float32)

        for ie in range(o.batchsz): # batchsz
            # randomly select an object
            objid = random.sample(self.objids_valid_snp[dstype][idx[ie]], 1)[0]

            # randomly select segment of frames (t<T)
            frms = select_frms(self.objvalidfrms_snp[dstype][idx[ie]][objid])

            for t, frm in enumerate(frms):
                # for x; image
                fimg = self.snps[dstype]['Data'][idx[ie]] + '/{0:06d}.JPEG'.format(frm)
                x = cv2.imread(fimg)[:,:,(2,1,0)]

                # for y; label
                xmlfile = self.snps[dstype]['Annotations'][idx[ie]] + '/{0:06d}.xml'.format(frm)
                y = get_bndbox_from_xml(xmlfile, objid)

                # image resize. NOTE: the best image size? need experiments
                if not o.useresizedimg: 
                    data[ie,t] = cv2.resize(x, (o.frmsz, o.frmsz), 
                        interpolation=cv2.INTER_AREA)
                else:
                    data[ie,t] = x
                label[ie,t] = y
            #inputs_valid[ie,:len(frms)] = 1 # NOTE: currently not sparse sequence
            inputs_length[ie] = len(frms)
            inputs_HW[ie] = x.shape[0:2]

        # TODO: Data augmentation
        # 1. data augmentation (rotation, scaling, translation)
        # 2. data perturbation.. 

        # image normalization 
        data -= self.stat[dstype]['mean']
        data /= self.stat[dstype]['std']

        batch = {
                'inputs': data,
                #'inputs_valid': inputs_valid, 
                'inputs_length': inputs_length, 
                'inputs_HW': inputs_HW,
                'labels': label,
                'idx': idx
                }

        return batch
    
    def update_epoch_begin(self, dstype):
        '''Perform updates at each epoch. Whatever necessary comes into this.
        '''
        # may no need dstype, as this shuffles train data only.
        self.idx_shuffle[dstype] = np.random.permutation(self.nexps[dstype])

    def run_sanitycheck(self, batch, dataset, frmsz):
        draw.show_dataset_batch(batch, dataset, frmsz, self.stat[dstype])

    def save_resized_images(self, o):
        '''
        This module performs resizing of images offline. 
        '''
        assert(o.trainsplit == 9)
        assert(o.useresizedimg == False)
        for dstype in ['train', 'val']: # NOTE: need to perform for test set
            for i, snp in enumerate(self.snps[dstype]['Data']):
                print dstype, i, self.nsnps[dstype], snp

                savedir = snp.replace('Data', 'Data_frmsz{}'.format(o.frmsz))
                if not os.path.exists(savedir): helpers.mkdir_p(savedir)

                imglist = glob.glob(snp+'/*.JPEG')
                for img in imglist:
                    # image read and resize
                    x = cv2.resize(cv2.imread(img), 
                            (o.frmsz, o.frmsz), interpolation=cv2.INTER_AREA)
                    # save
                    fname = img.replace('Data', 'Data_frmsz{}'.format(o.frmsz))
                    cv2.imwrite(fname, x)


def load_data(o):
    if o.dataset == 'moving_mnist':
        loader = Data_moving_mnist(o)
    elif o.dataset == 'bouncing_mnist':
        loader = Data_bouncing_mnist(o)
    elif o.dataset == 'ILSVRC':
        loader = Data_ILSVRC(o)
    else:
        raise ValueError('dataset not implemented yet')
    return loader

if __name__ == '__main__':

    # settings 
    np.random.seed(9)

    from opts import Opts
    o = Opts()
    o.batchsz = 20
    o.dataset = 'ILSVRC' # moving_mnist, bouncing_mnist, ILSVRC 
    o.trainsplit = 0
    o._set_dataset_params()
    dstype = 'train'
    
    #o.useresizedimg = False # NOTE: set it False if need to create resized imgs
    loader = load_data(o)
    batch = loader.get_batch(0, o, dstype)
    #loader.run_sanitycheck(batch, o.dataset, o.frmsz)
    #loader.save_resized_images(o) # to create resized images
    pdb.set_trace()

