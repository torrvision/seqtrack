'''Describes different datasets.

A dataset is a collection of videos,
where each video contains a collection of tracks.
Each track may have the location of the object specified in any subset of frames.
The location of the object is a rectangle.
There is currently no distinction between an object being absent versus unlabelled.

A dataset object has the following properties:

    dataset.videos -- List of strings.
    dataset.video_length[video] -- Dictionary that maps string -> integer.
    dataset.image_file(video, frame_number) -- Returns absolute path.
    dataset.image_size[video] -- Dictionary that maps string -> (width, height).
    dataset.original_image_size[video] -- Dictionary that maps string -> (width, height).
        Used for computing IOU, distance, etc.
    dataset.tracks[video] -- Dictionary that maps string -> list of tracks.
        A track is a dictionary that maps frame_number -> rectangle.
        The subset of keys indicates in which frames the object is labelled.
        The rectangle is in the form [xmin, ymin, xmax, ymax].
        The rectangle coordinates are normalized to [0, 1].
'''

import pdb
import numpy as np
import cPickle as pickle
import gzip
import json

# packages for bouncing mnist
import h5py
import scipy.ndimage as spn
import os

import glob
import itertools
import random
import xmltodict
import cv2
from PIL import Image

import draw
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
        inputs_valid = np.ones((o.batchsz, o.ntimesteps+1), dtype=np.bool)

        inputs_HW = np.ones((o.batchsz, 2), dtype=np.float32) * self.frmsz

        # Add fixed sized window to make label 4 dimension. [only moving_mnist]
        pos = np.concatenate((pos, pos+28), axis=2)

        # relative scale of label
        pos = pos / float(o.frmsz)

        # add one more dimension to data for placeholder tensor shape
        vids = np.expand_dims(vids, axis=4)

        batch = {
                'inputs': vids,
                'inputs_valid': inputs_valid, 
                'inputs_HW': inputs_HW, 
                'labels': pos,
                'digits': data['targets'][idx],
                'idx': idx
                }
        return batch

    def update_epoch_begin(self, dstype):
        # may no need dstype, as this shuffles train data only.
        self.idx_shuffle[dstype] = np.random.permutation(self.nexps[dstype])


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
        inputs_valid = np.ones((o.batchsz, o.ntimesteps+1), dtype=np.bool)

        inputs_HW = np.ones((o.batchsz, 2), dtype=np.float32) * self.frmsz

        # relative scale of label
        label = label / o.frmsz

        # add one more dimension to data for placeholder tensor shape
        data = np.expand_dims(data, axis=4)

        batch = {
                'inputs': data,
                'inputs_valid': inputs_valid, 
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


class Data_ILSVRC(object):
    def __init__(self, dstype, o):
        self.dstype      = dstype
        self.path_data   = o.path_data
        self.trainsplit  = o.trainsplit # 0, 1, 2, 3, or 9 for all
        self.datadirname = 'Data_frmsz{}'.format(o.frmsz) \
                                if o.useresizedimg else 'Data'

        self.videos = None
        # Number of frames in each snippet.
        self.video_length = None
        self.tracks = None
        self.stat = None
        self._load_data(o)

    def _images_dir(self, video):
        parts = video.split('/')
        return os.path.join(self.path_data, self.datadirname, 'VID', self.dstype, *parts)

    def _annotations_dir(self, video):
        parts = video.split('/')
        return os.path.join(self.path_data, 'Annotations', 'VID', self.dstype, *parts)

    def image_file(self, video, frame):
        return os.path.join(self._images_dir(video), '{:06d}.JPEG'.format(frame))

    def _load_data(self, o):
        # TODO: need to process and load test data set as well..
        # TODO: not loading test data yet. ILSVRC doesn't have "Annotations" for
        # test set. I can still load "Data", but need to fix a couple of places
        # to load only "Data". Will fix it later.
        self._update_videos()
        self._update_video_length()
        self._update_tracks(o)
        self._update_stat(o)

    def _parsexml(self, xmlfile):
        with open(xmlfile) as f:
            doc = xmltodict.parse(f.read(), force_list={'object'})
        return doc

    def _update_videos(self):
        def create_videos():
            if self.dstype in {'test', 'val'}:
                path_snp_data = os.path.join(self.path_data, self.datadirname, 'VID', self.dstype)
                videos = sorted(os.listdir(path_snp_data))
            elif self.dstype is 'train': # train data has 4 splits.
                splits = []
                if self.trainsplit in range(4):
                    splits = [self.trainsplit]
                elif self.trainsplit == 9: # MAGIC NUMBER for all train data 
                    splits = range(4)
                else:
                    raise ValueError('No available option for train split')
                videos = []
                for i in splits:
                    split_dir = 'ILSVRC2015_VID_train_{:04d}'.format(i)
                    path_snp_data = os.path.join(self.path_data, self.datadirname, 'VID', self.dstype, split_dir)
                    vs = sorted(os.listdir(path_snp_data))
                    vs = ['{}/{}'.format(split_dir, v) for v in vs]
                    videos.extend(vs)
            return videos

        if self.videos is None:
            self.videos = create_videos()

    def _update_video_length(self):
        def create_video_length():
            nfrms = {}
            for video in self.videos:
                images = glob.glob(os.path.join(self._images_dir(video), '*.JPEG'))
                nfrms[video] = len(images)
            return nfrms

        if self.video_length is None:
            self.video_length = create_video_length()

    def _identifier(self):
        if self.dstype == 'train':
            return '{}_{}'.format(self.dstype, self.trainsplit)
        else:
            return self.dstype

    def _update_tracks(self, o):
        def load_info():
            info = {}
            for i, video in enumerate(self.videos):
                print i+1, video
                video_info = load_video_info(video)
                for k, v in video_info.iteritems():
                    info.setdefault(k, {})[video] = v
            return info

        def load_video_info(video):
            frames = []
            size = None
            video_len = self.video_length[video]
            for t in range(video_len):
                xmlfile = os.path.join(self._annotations_dir(video), '{:06d}.xml'.format(t))
                frame_objects, frame_size = read_frame_annotation(xmlfile)
                if size is not None:
                    assert(size == frame_size)
                frames.append(frame_objects)
                size = size or frame_size
            # Convert from list of frame annotations to list of tracks.
            tracks = {}
            for t, frame_objects in enumerate(frames):
                for obj_id, rect in frame_objects.iteritems():
                    # List of (frame, rectangle) pairs instead of dictionary
                    # because JSON does not support integer keys.
                    tracks.setdefault(obj_id, []).append((t, rect))
                    # tracks.setdefault(obj_id, {})[t] = rect
            # Note: This will not preserve the original object IDs
            # if the IDs are not consecutive starting from 0.
            # For example: ILSVRC2015_VID_train_0000/ILSVRC2015_train_00014017
            return {
                'tracks': [tracks[obj_id] for obj_id in sorted(tracks.keys())],
                'original_image_size': size,
            }

        def read_frame_annotation(xmlfile):
            '''Returns a dictionary of (track index) -> rectangle.'''
            doc = self._parsexml(xmlfile)
            width  = int(doc['annotation']['size']['width'])
            height = int(doc['annotation']['size']['height'])
            obj_annots = doc['annotation'].get('object', [])
            objs = dict(map(lambda obj: extract_id_rect_pair(obj, width, height),
                            obj_annots))
            return objs, (width, height)

        def extract_id_rect_pair(obj, width, height):
            index = int(obj['trackid'])
            # ImageNet uses range xmin <= x < xmax.
            rect = [
                int(obj['bndbox']['xmin']) / float(width),
                int(obj['bndbox']['ymin']) / float(height),
                int(obj['bndbox']['xmax']) / float(width),
                int(obj['bndbox']['ymax']) / float(height),
            ]
            return index, rect

        if self.tracks is None:
            if not os.path.isdir(o.path_aux):
                os.makedirs(o.path_aux)
            cache_file = os.path.join(o.path_aux, 'info_{}.json'.format(self._identifier()))
            info = helpers.cache_json(cache_file, lambda: load_info())
            # Convert (frame, rectangle) pairs to dictionary.
            self.tracks = {video: [dict(track) for track in track_list]
                           for video, track_list in info['tracks'].iteritems()}
            self.original_image_size = info['original_image_size']

    def _update_stat(self, o):
        def create_stat_pixelwise(): # NOTE: obsolete
            stat = dict.fromkeys({'mean', 'std'}, None)
            mean = []
            std = []
            for i, video in enumerate(self.videos):
                print 'computing mean and std in snippet of {}, {}/{}'.format(
                        self.dstype, i+1, len(self.videos))
                imglist = sorted(glob.glob(os.path.join(self._images_dir(video), '*.JPEG')))
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

        def create_stat_global():
            stat = dict.fromkeys({'mean', 'std'}, None)
            means = []
            stds = []
            for i, video in enumerate(self.videos):
                print 'computing mean and std in snippet of {}, {}/{}'.format(
                        self.dstype, i+1, len(self.videos))
                imglist = sorted(glob.glob(os.path.join(self._images_dir(video), '*.JPEG')))
                xs = []
                # for j in imglist:
                #     # NOTE: perform resize image!
                #     xs.append(cv2.resize(cv2.imread(j)[:,:,(2,1,0)], 
                #         (o.frmsz, o.frmsz), interpolation=cv2.INTER_AREA))
                # Lazy method: Read a single image, do not resize.
                xs.append(cv2.imread(imglist[len(imglist)/2])[:,:,(2,1,0)])
                means.append(np.mean(xs))
                stds.append(np.std(xs))
            mean = np.mean(means)
            std = np.mean(stds)
            stat['mean'] = mean
            stat['std'] = std
            return stat

        if self.stat is None:
            if self.dstype == 'train':
                filename = os.path.join(o.path_stat, 
                    'meanstd_{}_frmsz_{}_train_{}.npy'.format(o.dataset, o.frmsz, self.trainsplit))
            else:
                filename = os.path.join(o.path_stat,
                    'meanstd_{}_frmsz_{}_{}.npy'.format(o.dataset, o.frmsz, self.dstype))
            if os.path.exists(filename):
                self.stat = np.load(filename).tolist()
            else:
                self.stat = create_stat_global() 
                np.save(filename, self.stat)


class Data_OTB(object):
    '''Represents OTB-50 or OTB-100 dataset.'''

    def __init__(self, variant, o):
        self.path_data = os.path.join(o.path_data_home, 'OTB')
        self.variant = variant # {'OTB-50', 'OTB-100'}
        # Options to save locally.
        self.useresizedimg = o.useresizedimg
        self.frmsz         = o.frmsz

        # TODO: OTB dataset has attributes to videos. Add them.

        self.videos       = None
        self.video_length = None
        self.tracks       = None
        self._load_data(o)

    def image_file(self, video, t):
        if self.useresizedimg:
            img_dir = 'img'
        else:
            img_dir = 'img_frmsz{}'.format(self.frmsz)
        if video == 'Board':
            filename = '{:05d}.jpg'.format(t+1)
        else:
            filename = '{:04d}.jpg'.format(t+1)
        return os.path.join(self.path_data, video, img_dir, filename)

    def _load_data(self, o):
        self._update_videos()
        self._update_exceptions(o)
        self._update_video_info(o)

    def _update_videos(self):
        OTB50 = [
            'Basketball','Biker','Bird1','BlurBody','BlurCar2','BlurFace',
            'BlurOwl','Bolt','Box','Car1','Car4','CarDark','CarScale',
            'ClifBar','Couple','Crowds','David','Deer','Diving','DragonBaby',
            'Dudek','Football','Freeman4','Girl','Human3','Human4','Human6',
            'Human9','Ironman','Jump','Jumping','Liquor','Matrix',
            'MotorRolling','Panda','RedTeam','Shaking','Singer2','Skating1',
            'Skating2','Skiing','Soccer','Surfer','Sylvester',
            'Tiger2','Trellis','Walking','Walking2','Woman']
        OTB100 = OTB50 + [
            'Bird2','BlurCar1','BlurCar3','BlurCar4','Board','Bolt2','Boy', 
            'Car2','Car24','Coke','Coupon','Crossing','Dancer','Dancer2',
            'David2','David3','Dog','Dog1','Doll','FaceOcc1','FaceOcc2','Fish', 
            'FleetFace','Football1','Freeman1','Freeman3','Girl2','Gym',
            'Human2','Human5','Human7','Human8','Jogging',
            'KiteSurf','Lemming','Man','Mhyang','MountainBike','Rubik',
            'Singer1','Skater','Skater2','Subway','Suv','Tiger1','Toy','Trans', 
            'Twinnings','Vase']
        if self.variant == 'OTB-50':
            self.videos = OTB50
        elif self.variant == 'OTB-100':
            self.videos = OTB100
        else:
            raise ValueError('unknown OTB variant: {}'.format(self.variant))

    def _update_exceptions(self, o):
        # Note: First frame is indexed from 1!
        self.offset = {}
        self.offset['David']    = 300
        self.offset['BlurCar1'] = 247
        self.offset['BlurCar3'] = 3
        self.offset['BlurCar4'] = 18

    def _update_video_info(self, o):
        def load_info():
            info = {}
            for i, video in enumerate(self.videos):
                video_info = load_video_info(video)
                for k, v in video_info.iteritems():
                    info.setdefault(k, {})[video] = v
            return info

        def load_video_info(video):
            video_dir = os.path.join(self.path_data, video)
            if not os.path.isdir(video_dir):
                raise ValueError('video directory does not exist: {}'.format(video_dir))
            # Get video length.
            image_files = glob.glob(os.path.join(video_dir, 'img', '*.jpg'))
            video_length = len(image_files)
            assert(video_length > 0)
            # Check image resolution.
            width, height = Image.open(image_files[0]).size

            # Get number of objects.
            gt_files = glob.glob(os.path.join(video_dir, 'groundtruth_rect*.txt'))
            num_objects = len(gt_files)
            assert(num_objects > 0)
            tracks = []
            for j in range(num_objects):
                if num_objects == 1:
                    gt_file = 'groundtruth_rect.txt'
                else:
                    gt_file = 'groundtruth_rect.{}.txt'.format(j+1)
                rects = load_rectangles(os.path.join(video_dir, gt_file))
                if len(rects) == 0:
                    continue
                # Normalize by image size.
                rects = rects / np.array([width, height, width, height])
                # Create track from rectangles and offset.
                offset = self.offset.get(video, 1) - 1
                track = {t+offset: list(rect) for t, rect in enumerate(rects)}
                tracks.append(track)
            return {
                'video_length':        video_length,
                'original_image_size': (width, height),
                'tracks':              tracks,
            }

        def load_rectangles(fname):
            try:
                rects = np.loadtxt(fname, dtype=np.float32, delimiter=',')
            except:
                rects = np.loadtxt(fname, dtype=np.float32)
            if len(rects) == 0:
                return []
            # label reformat [x1,y1,w,h] -> [x1,y1,x2,y2]
            rects[:,(2,3)] = rects[:,(0,1)] + rects[:,(2,3)]
            # Assume rectangles are meant for use with Matlab's imshow() and rectangle().
            # Matlab draws an image of size n on the continuous range 0.5 to n+0.5.
            return rects - 0.5

        if self.tracks is None:
            info = load_info()
            self.video_length        = info['video_length']
            self.original_image_size = info['original_image_size']
            self.tracks              = info['tracks']


def get_masks_from_rectangles(rec, o):
    # create mask using rec; typically rec=y_prev
    x1 = rec[:,0] * o.frmsz
    y1 = rec[:,1] * o.frmsz
    x2 = rec[:,2] * o.frmsz
    y2 = rec[:,3] * o.frmsz
    grid_x, grid_y = np.meshgrid(np.arange(o.frmsz), np.arange(o.frmsz))
    # resize tensors so that they can be compared
    x1 = np.expand_dims(np.expand_dims(x1,1),2)
    x2 = np.expand_dims(np.expand_dims(x2,1),2)
    y1 = np.expand_dims(np.expand_dims(y1,1),2)
    y2 = np.expand_dims(np.expand_dims(y2,1),2)
    grid_x = np.tile(np.expand_dims(grid_x,0), [o.batchsz,1,1])
    grid_y = np.tile(np.expand_dims(grid_y,0), [o.batchsz,1,1])
    # mask
    masks = np.logical_and(
        np.logical_and(np.less_equal(x1, grid_x), 
            np.less_equal(grid_x, x2)),
        np.logical_and(np.less_equal(y1, grid_y), 
            np.less_equal(grid_y, y2)))
    # type and dim change so that it can be concated with x (add channel dim)
    masks = np.expand_dims(masks.astype(np.float32),3)
    return masks

def run_sanitycheck(batch, dataset, frmsz, stat=None, fulllen=False):
    if not fulllen:
        draw.show_dataset_batch(batch, dataset, frmsz, stat)
    else:
        draw.show_dataset_batch_fulllen_seq(batch, dataset, frmsz, stat)

def split_batch_fulllen_seq(batch_fl, o):
    # split the full-length sequence in multiple segments
    nsegments = int(np.ceil( (batch_fl['nfrms']-1)/float(o.ntimesteps) ))

    data = np.zeros(
            (nsegments, o.ntimesteps+1, o.frmsz, o.frmsz, o.ninchannel), 
            dtype=np.float32)
    label = np.zeros((nsegments, o.ntimesteps+1, o.outdim), 
            dtype=np.float32)
    inputs_valid = np.zeros((nsegments, o.ntimesteps+1), dtype=np.bool)
    inputs_HW = np.zeros((nsegments, 2), dtype=np.float32)

    for i in range(nsegments):
        if (i+1)*o.ntimesteps+1 <= batch_fl['nfrms']:
            seglen = o.ntimesteps + 1
        else:
            seglen = (batch_fl['nfrms']-1) % o.ntimesteps + 1
        data[i, 0:seglen] = batch_fl['inputs'][
                0, i*o.ntimesteps:i*o.ntimesteps + seglen]
        label[i, 0:seglen] = batch_fl['labels'][
                0, i*o.ntimesteps:i*o.ntimesteps + seglen]
        inputs_valid[i, 0:seglen] = batch_fl['inputs_valid'][
                0, i*o.ntimesteps:i*o.ntimesteps + seglen]
        inputs_HW[i] = batch_fl['inputs_HW'][0]

    batch = {
            'inputs': data,
            'inputs_valid': inputs_valid,
            'inputs_HW': inputs_HW,
            'labels': label,
            'nfrms': batch_fl['nfrms'],
            'nsegments': nsegments,
            'idx': batch_fl['idx']
            }
    return batch 


def load_data(o):
    if o.dataset == 'ILSVRC':
        datasets = {dstype: Data_ILSVRC(dstype, o) for dstype in ['train', 'val']}
    else:
        raise ValueError('dataset not implemented yet')
    return datasets

if __name__ == '__main__':

    # settings 
    np.random.seed(9)

    from opts import Opts
    o = Opts()
    o.batchsz = 10
    o.ntimesteps = 20

    o.dataset = 'ILSVRC' # moving_mnist, bouncing_mnist, ILSVRC, OTB-50
    o._set_dataset_params()

    dstype = 'train'
    
    test_fl = False

    sanitycheck = False

    # moving_mnist
    if o.dataset in ['moving_mnist', 'bouncing_mnist']:
        loader = load_data(o)
        batch = loader.get_batch(0, o, dstype)
        if sanitycheck: run_sanitycheck(batch, o.dataset, o.frmsz)

    # ILSVRC
    if o.dataset == 'ILSVRC':
        o.trainsplit = 0
        loader = load_data(o)
        if test_fl: # to test full-length sequences 
            batch_fl = loader.get_batch_fl(0, o)
            batch = split_batch_fulllen_seq(batch_fl, o)
            if sanitycheck: run_sanitycheck(batch, o.dataset, o.frmsz, loader.stat[dstype], fulllen=True) 
        else:
            batch = loader.get_batch(0, o, dstype)
            if sanitycheck: run_sanitycheck(batch, o.dataset, o.frmsz, loader.stat[dstype])
        #o.useresizedimg = False # NOTE: set it False to create resized imgs
        #loader.create_resized_images(o) # to create resized images

    # OTB-50
    if o.dataset in ['OTB-50', 'OTB-100']:
        loader = load_data(o)
        assert(test_fl==True) # for OTB, it's always full-length
        batch_fl = loader.get_batch_fl(0, o)
        batch = split_batch_fulllen_seq(batch_fl, o)
        if sanitycheck: run_sanitycheck(batch, o.dataset, o.frmsz, loader.stat, fulllen=True)
        #o.useresizedimg = False # NOTE: set it False to create resized imgs
        #loader.create_resized_images(o)

    pdb.set_trace()

