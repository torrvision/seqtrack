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


class Data_moving_mnist(object):
    '''
    The main routine to generate moving mnist is derived from 
    "RATM: Recurrent Attentive Tracking Model"
    '''
    def __init__(self, o):
        self.datafile = o.path_data+'/'+o.dataset+'/mnist.pkl.gz' 
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

    def get_batch(self, ib, o, dstype):
        '''
        Everytime this function is called, create the batch number of moving 
        mnist sets. If this process has randomness, no other data augmentation 
        technique is applied for now. 
        '''
        data = self.data[dstype]
        idx = self.idx_shuffle[dstype][(ib*o.batchsz):(ib+1)*o.batchsz] 

        # the following is a modified version from RATM data preparation
        vids = np.zeros((o.batchsz, o.ntimesteps, self.frmsz, self.frmsz), 
                dtype=np.float32)
        pos_init = np.random.randint(self.frmsz-28, size=(o.batchsz,2))
        pos = np.zeros((o.batchsz, o.ntimesteps, 2), dtype=np.float32)
        pos[:,0] = pos_init

        posmax = self.frmsz-29

        d = np.random.randint(low=-15, high=15, size=(o.batchsz,2))

        for t in range(o.ntimesteps):
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

        # Add fixed sized window to make label 4 dimension. [only moving_mnist]
        pos = np.concatenate((pos, pos+28), axis=2)

        # relative scale of label
        pos = pos / o.frmsz

        # add one more dimension to data for placeholder tensor shape
        vids = np.expand_dims(vids, axis=4)

        batch = {
                'inputs': vids,
                'inputs_length': inputs_length, 
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
        self.data_path = os.path.join(o.path_base, 'data/bouncing_mnist')
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
        self.image_size_ = 100 
        self.scale_range = 0.1 
        self.buff = True
        self.step_length_ = 0.1
        self.digit_size_ = 28
        self.frame_size_ = self.image_size_ ** 2
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
        self.buff_data = np.zeros((self.buff_size, o.ntimesteps, 
            self.image_size_, self.image_size_), dtype=np.float32)
        self.buff_label = np.zeros((self.buff_size, o.ntimesteps, 4))
        self.clutter_move = 1 
        self.with_clutters = 1 

    def _load_data(self):
        f = h5py.File(os.path.join(self.data_path, 'mnist.h5'))
        # f has 'train' and 'test' and each has 'inputs' and 'targets'
        for dstype in ['train', 'test']:
            self.data[dstype] = dict.fromkeys({'images', 'targets'}, None) 
            self.data[dstype]['images'] = np.asarray(f['train/inputs'].value)
            self.data[dstype]['targets'] = np.asarray(f['train/targets'].value)
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
            image_size_ = self.image_size_
        if object_size_ is None:
            object_size_ = self.digit_size_
        if step_length_ is None:
            step_length_ = self.step_length_
        length = o.ntimesteps
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
            image_size_ = self.image_size_ * 2
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
            image_size_ = self.image_size_
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

    def get_batch(self, ib, o, dstype, verbose=False, count=1):
        '''Here in this function also made several changes in several places.
        '''
        data_all = self.data[dstype]
        idx = self.idx_shuffle[dstype][(ib*o.batchsz):(ib+1)*o.batchsz] 
        
        start_y, start_x = self._GetRandomTrajectory(o.batchsz * self.num_digits_, o)
        window_y, window_x = self._GetRandomTrajectory(o.batchsz * 1, o, self.image_size_*2, object_size_=self.image_size_, step_length_ = 1e-2)
        # TODO: change data to real image or cluttered background
        data = np.zeros((o.batchsz, o.ntimesteps, self.image_size_, self.image_size_), dtype=np.float32)
        label = np.zeros((o.batchsz, o.ntimesteps, 4), dtype=np.float32)

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
                        for i in range(o.ntimesteps):
                            wx = window_x[i,j]
                            wy = window_y[i,j]
                            data[j, i] = self._Overlap(clutter_bg[wy:wy+self.image_size_, wx:wx+self.image_size_], data[j, i])
                    else:
                        for i in range(o.ntimesteps):
                            wx = window_x[0, j]
                            wy = window_y[0, j]
                            data[j, i] = self._Overlap(clutter_bg[wy:wy+self.image_size_, wx:wx+self.image_size_], data[j, i])
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
                    for i in range(o.ntimesteps):
                        scale_factor = np.exp((np.random.random_sample()-0.5)*self.scale_range)
                        scale_image = spn.zoom(digit_image, scale_factor)
                        digit_size_ = digit_size_ * scale_factor 
                        top    = start_y[i, j * self.num_digits_ + n]
                        left   = start_x[i, j * self.num_digits_ + n]
                        if digit_size_!=np.shape(scale_image)[0]:
                            digit_size_ = np.shape(scale_image)[0]
                        bottom = top  + digit_size_
                        right  = left + digit_size_
                        if right>self.image_size_ or bottom>self.image_size_:
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
                        data[j, i] = self._Overlap(data[j, i], clutter[wy:wy+self.image_size_, wx:wx+self.image_size_])
                        # NL: (y,x) -> (x,y)
                        #label[j, i] = label_offset + np.array([top, left, top, left])
                        label[j, i] = label_offset + np.array([left, top, left, top])
                if wc:
                    if cm:
                        for i in range(o.ntimesteps):
                            wx = window_x[i,j]
                            wy = window_y[i,j]
                            data[j, i] = self._Overlap(data[j, i], clutter[wy:wy+self.image_size_, wx:wx+self.image_size_])
                    else:
                        for i in range(o.ntimesteps):
                            wx = window_x[0,j]
                            wy = window_y[0,j]
                            data[j, i] = self._Overlap(data[j, i], clutter[wy:wy+self.image_size_, wx:wx+self.image_size_])
                if self.buff:
                    self._setBuff(data[j], label[j])

        # TODO: variable length inputs for
        # 1. online learning, 2. arbitrary training sequences 
        inputs_length = np.ones((o.batchsz), dtype=np.int32) * o.ntimesteps

        # relative scale of label
        label = label / o.frmsz

        # add one more dimension to data for placeholder tensor shape
        data = np.expand_dims(data, axis=4)

        batch = {
                'inputs': data,
                'inputs_length': inputs_length, 
                'labels': label,
                'digits': data_all['targets'][idx],
                'idx': idx
                }

        return batch

    def update_epoch_begin(self, dstype):
        # may no need dstype, as this shuffles train data only.
        self.idx_shuffle[dstype] = np.random.permutation(self.nexps[dstype])

    def run_sanitycheck(self, batch, dataset, frmsz):
        draw.show_dataset_batch(batch, dataset, frmsz)


class Data_ilsvrc(object):
    def __init__(self, o):
        self.path_data = '/home/namhoon/data/ILSVRC' # TODO: loadable in server

        self.snp            = dict.fromkeys({'train', 'val', 'test'}, None)
        self.nsnps          = dict.fromkeys({'train', 'val', 'test'}, None)
        self.nfrms_snp      = dict.fromkeys({'train', 'val', 'test'}, None)
        self.objids_snp     = dict.fromkeys({'train', 'val', 'test'}, None)
        self.maxtrackid_snp = dict.fromkeys({'train', 'val', 'test'}, None)
        self.snp_frmsplits  = dict.fromkeys({'train', 'val', 'test'}, None)

        self.exps           = dict.fromkeys({'train', 'val', 'test'}, None)
        self.nexps          = dict.fromkeys({'train', 'val', 'test'}, None)
        self.idx_shuffle    = dict.fromkeys({'train', 'val', 'test'}, None)

        self._load_data(o)
        self._update_idx_shuffle(['train', 'val', 'test'])

    def _load_data(self, o):
        # TODO: not loading test data yet. ILSVRC doesn't have "Annotations" for
        # test set. I can still load "Data", but need to fix a couple of places
        # to load only "Data". Will fix it later.
        for dstype in ['train', 'val']: # ['train', 'val', 'test']:
            self._update_snps(dstype)
            self._update_nsnps(dstype)
            self._update_nfrms_snp(dstype)
            self._update_objids_snp(dstype, o)
            self._update_maxtrackid_snp(dstype)
            self._update_snp_frmsplits(dstype, o.ntimesteps) # NOTE: perform at every epoch?
            self._update_exps(dstype)
            self._update_nexps(dstype)

    def _parsexml(self, xmlfile):
        with open(xmlfile) as f:
            doc = xmltodict.parse(f.read())
        return doc

    def _update_snps(self, dstype):
        def create_snps(dstype):
            output = {}
            if dstype is 'test':
                xory = ['Data'] # no annotations for test data
            else:
                xory = ['Annotations', 'Data']
            for i, val in enumerate(xory):
                path_snp = os.path.join(
                    self.path_data, '{}/VID/{}'.format(val, dstype))
                snps = os.listdir(path_snp)
                snps = [path_snp + '/' + snp for snp in snps]
                output[val] = snps
            return output
        if self.snp[dstype] is None:
            self.snp[dstype] = create_snps(dstype)

    def _update_nsnps(self, dstype):
        if self.nsnps[dstype] is None:
            self.nsnps[dstype] = len(self.snp[dstype]['Data'])

    def _update_nfrms_snp(self, dstype):
        def create_nfrms_snp(dstype):
            nfrms = []
            for snp in self.snp[dstype]['Data']:
                nfrms.append(len(glob.glob(snp+'/*.JPEG')))
            return nfrms
        if self.nfrms_snp[dstype] is None:
            self.nfrms_snp[dstype] = create_nfrms_snp(dstype)

    def _update_objids_snp(self, dstype, o):
        def extract_objids_snp(dstype):
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

            objids_snp = []
            for i in range(self.nsnps[dstype]):
                print i
                objids_frm = []
                for j in range(self.nfrms_snp[dstype][i]):
                    xmlfile = os.path.join(self.snp[dstype]['Annotations'][i], 
                        '{0:06d}.xml'.format(j))
                    objids_frm.append(extract_objids_from_xml(xmlfile))
                objids_snp.append(objids_frm)
            return objids_snp
    
        if self.objids_snp[dstype] is None:
            filename = os.path.join(
                o.path_aux, 'objids_snp_{}.npy'.format(dstype))
            if os.path.exists(filename):
                self.objids_snp[dstype] = np.load(filename).tolist()
            else: # if no file, create and also save
                self.objids_snp[dstype] = extract_objids_snp(dstype)
                np.save(filename, self.objids_snp[dstype])

    def _update_maxtrackid_snp(self, dstype):
        def compute_maxtrackid_snp(dstype):
            maxtrackid_snp = []
            for i in range(self.nsnps[dstype]):
                '''This is wrong; need max trackid, which is the number of total 
                objects in a snippet.
                currentmax = 0
                for j in self.objids_snp[dstype][i]:
                    if len(j) > currentmax:
                        currentmax = len(j)
                maxtrackid_snp.append(currentmax)
                '''
                currentmax = 0
                for j in self.objids_snp[dstype][i]:
                    if max(j) > currentmax:
                        currentmax = max(j)
                maxtrackid_snp.append(currentmax)
            return maxtrackid_snp 

        if self.maxtrackid_snp[dstype] is None:
            self.maxtrackid_snp[dstype] = compute_maxtrackid_snp(dstype)

    def _update_snp_frmsplits(self, dstype, frm_max):
        def create_snp_frmsplits(dstype, frm_max):
            def _get_split_sum(frm_min, frm_max, seq_length):
                cnt = 0
                lengths = []
                while cnt < seq_length:
                    if seq_length-cnt < frm_max:
                        length = seq_length-cnt
                    else:
                        length = np.random.randint(frm_min, frm_max)
                    lengths.append(length)
                    cnt += length
                    #print length, cnt, seq_length
                assert(np.sum(lengths)==seq_length)
                return lengths
            frm_min = 2
            output = {}
            for i, nfrms_snp in enumerate(self.nfrms_snp[dstype]):
                output[i] = _get_split_sum(frm_min, frm_max, nfrms_snp)
            return output

        if self.snp_frmsplits[dstype] is None:
            self.snp_frmsplits[dstype] = create_snp_frmsplits(dstype, frm_max)

    def _update_exps(self, dstype):
        def create_exps(dstype):
            # NOTE: Since dataset is big, I can't contain actual 'data' in one 
            # dict. Instead, actual data will be loaded during batch loading.
            # The dict 'exps' contains necessary information to load such data.
            exps = {}
            exps['Annotation'] = []
            exps['Data'] = []
            exps['frm'] = []
            exps['trackid'] = []
            for i in range(self.nsnps[dstype]):
                frms = [0] + np.cumsum(self.snp_frmsplits[dstype][i]).tolist()
                # NOTE: creating sequences for every objects in the snippet.
                for iobj in range(self.maxtrackid_snp[dstype][i]+1):
                    for iseg in range(len(frms)-1):
                        exps['Annotation'].append(self.snp[dstype]['Annotations'][i])
                        exps['Data'].append(self.snp[dstype]['Data'][i])
                        exps['frm'].append(range(frms[iseg], frms[iseg+1]))
                        exps['trackid'].append(iobj)
            return exps

        if self.exps[dstype] is None:
            self.exps[dstype] = create_exps(dstype)

    def _update_nexps(self, dstype):
        if self.nexps[dstype] is None:
            self.nexps[dstype] = len(self.exps[dstype]['Data'])

    def _update_idx_shuffle(self, dstypes):
        for dstype in dstypes:
            if dstype in self.nexps and self.nexps[dstype] is not None:
                self.idx_shuffle[dstype] = np.random.permutation(self.nexps[dstype])
    
    def get_batch(self, ib, o, dstype):
        def get_bndbox_from_xml(xmlfile, trackid):
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
                        if int(doc['annotation']['object'][i]['trackid']) == trackid:
                            bndbox = [
                                np.float32(doc['annotation']['object'][i]['bndbox']['xmin']) / w,
                                np.float32(doc['annotation']['object'][i]['bndbox']['ymin']) / h,
                                np.float32(doc['annotation']['object'][i]['bndbox']['xmax']) / w,
                                np.float32(doc['annotation']['object'][i]['bndbox']['ymax']) / h]
                            break
                else:
                    if int(doc['annotation']['object']['trackid']) == trackid:
                        bndbox = [
                            np.float32(doc['annotation']['object']['bndbox']['xmin']) / w,
                            np.float32(doc['annotation']['object']['bndbox']['ymin']) / h,
                            np.float32(doc['annotation']['object']['bndbox']['xmax']) / w,
                            np.float32(doc['annotation']['object']['bndbox']['ymax']) / h]
            return bndbox # xyxy format

        idx = self.idx_shuffle[dstype][(ib*o.batchsz):(ib+1)*o.batchsz] 

        data = np.zeros(
            (o.batchsz, o.ntimesteps, o.frmsz, o.frmsz, o.ninchannel), 
            dtype=np.float32)
        label = np.zeros((o.batchsz, o.ntimesteps, o.outdim), dtype=np.float32)
        inputs_length = np.zeros((o.batchsz), dtype=np.int32)

        for ie in range(o.batchsz):
            for t, ifrm in enumerate(self.exps[dstype]['frm'][idx[ie]]):
                # for x; image
                fimg = self.exps[dstype]['Data'][idx[ie]] \
                    + '/{0:06d}.JPEG'.format(ifrm)
                x = cv2.imread(fimg)[:,:,(2,1,0)]
                
                # for y; label
                xmlfile = self.exps[dstype]['Annotation'][idx[ie]] \
                    + '/{0:06d}.xml'.format(ifrm)
                trackid = self.exps[dstype]['trackid'][idx[ie]]
                # NOTE: every example has an assigned trackid, but this doesn't
                # mean that it has the object at every frame. In case it doesn't
                # have that object, y is assigned [None, None, None, None]. 
                # This will be converted to nan when saving into a numpy array.
                y = get_bndbox_from_xml(xmlfile, trackid)

                # image resize. NOTE: the best image size? need experiments
                data[ie,t] = cv2.resize(x, (o.frmsz, o.frmsz), 
                    interpolation=cv2.INTER_AREA)
                label[ie,t] = y
            inputs_length[ie] = t+1

        # TODO: 
        # 1. image normalization with mean and std
        # 2. data augmentation (rotation, scaling, translation)
        # 3. data perturbation.. (need to think about this)

        batch = {
                'inputs': data,
                'inputs_length': inputs_length, 
                'labels': label,
                'idx': idx
                }

        return batch

    def update_epoch_begin(self, dstype):
        '''Perform updates at each epoch. Whatever necessary comes into this.
        '''
        # may no need dstype, as this shuffles train data only.
        self.idx_shuffle[dstype] = np.random.permutation(self.nexps[dstype])
        # NOTE: consider updating random splits
        # If this is the case, 'nbatch' in train module should be adaptive.
        #self._update_snp_frmsplits(self, dstype, frm_max):

    def run_sanitycheck(self, batch, dataset, frmsz):
        draw.show_dataset_batch(batch, dataset, frmsz)


def load_data(o):
    if o.dataset == 'moving_mnist':
        loader = Data_moving_mnist(o)
    elif o.dataset == 'bouncing_mnist':
        loader = Data_bouncing_mnist(o)
    elif o.dataset == 'ilsvrc':
        loader = Data_ilsvrc(o)
    else:
        raise ValueError('dataset not implemented yet')
    return loader

if __name__ == '__main__':

    # settings 
    np.random.seed(9)

    from opts import Opts
    o = Opts()
    o.batchsz = 20
    o.dataset = 'ilsvrc' # moving_mnist, bouncing_mnist, ilsvrc
    o._set_dataset_params()
    dstype = 'train'
    loader = load_data(o)
    batch = loader.get_batch(0, o, dstype)
    #loader.run_sanitycheck(batch, o.dataset, o.frmsz)
    pdb.set_trace()

