import pdb
import numpy as np
import cPickle as pickle
import gzip

# packages for bouncing mnist
import h5py
import scipy.ndimage as spn
import os

import draw


class Data_moving_mnist(object):
    '''
    The main routine to generate moving mnist is derived from 
    "RATM: Recurrent Attentive Tracking Model"
    '''
    def __init__(self, o):
        self.datafile = o.path_data+'/'+o.dataset+'/mnist.pkl.gz' 
        self.frmsz = o.moving_mnist['frmsz']
        self.featdim = o.moving_mnist['featdim'] # TODO: change featdim (CNN)
        self.outdim = o.moving_mnist['outdim']

        self._data = None  
        self._data_tr, self._data_va, self.data_te = None, None, None
        self._load_data()
        self._split_data()
        assert self.data_tr['images'].shape[0]==self.data_tr['targets'].shape[0]
        assert self.data_va['images'].shape[0]==self.data_va['targets'].shape[0]
        assert self.data_te['images'].shape[0]==self.data_te['targets'].shape[0]
        self.ntr = self.data_tr['images'].shape[0]
        self.nva = self.data_va['images'].shape[0]
        self.nte = self.data_te['images'].shape[0]

        self.idx_shuffle_tr = np.random.permutation(self.ntr)
        self.idx_shuffle_va = np.random.permutation(self.nva)
        self.idx_shuffle_te = np.random.permutation(self.nte)

    def _load_data(self):
        with gzip.open(self.datafile, 'rb') as f:
             rawdata = pickle.load(f)
        self.data = {}
        for p, part in enumerate(('train', 'val', 'test')):
            self.data[part] = {}
            self.data[part]['images'] = rawdata[p][0].reshape(-1, 28, 28)
            self.data[part]['targets'] = rawdata[p][1]

    def _split_data(self):
        self.data_tr = self.data['train']
        self.data_va = self.data['val']
        self.data_te = self.data['test']
             
    def get_batch(self, ib, o, data_=None):
        '''
        Everytime this function is called, create the batch number of moving 
        mnist sets. If this process has randomness, no other data augmentation 
        technique is applied for now. 
        '''
        if data_ == 'train':
            data = self.data_tr
            idx_shuffle = self.idx_shuffle_tr
        elif data_ == 'val':
            data = self.data_va
            idx_shuffle = self.idx_shuffle_va
        elif data_ == 'test':
            data = self.data_te
            idx_shuffle = self.idx_shuffle_te
        else:
            raise ValueError('wrong data')

        # the following is a modified version from RATM data preparation
        vids = np.zeros((o.batchsz, o.ntimesteps, self.frmsz, self.frmsz))
        pos_init = np.random.randint(self.frmsz-28, size=(o.batchsz,2))
        pos = np.zeros((o.batchsz, o.ntimesteps, 2), dtype=np.int32)
        pos[:,0] = pos_init

        posmax = self.frmsz-29

        d = np.random.randint(low=-15, high=15, size=(o.batchsz,2))
        idx = idx_shuffle[ib*o.batchsz:(ib+1)*o.batchsz]

        for t in range(o.ntimesteps):
            dtm1 = d
            d = np.random.randint(low=-15, high=15, size=(o.batchsz,2))
            for i in range(o.batchsz):
                vids[i,t,
                        pos[i,t,0]:pos[i,t,0]+28,
                        pos[i,t,1]:pos[i,t,1]+28] = \
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

        # Add fixed sized window for addtional label; [only moving_mnist]
        pos = np.concatenate((pos, pos+28), axis=2)

        batch = {
                'inputs': vids,
                'inputs_length': inputs_length, 
                'labels': pos,
                'digits': data['targets'][idx],
                'idx': idx
                }
        return batch

    def run_sanitycheck(self, batch):
        draw.show_moving_mnist(batch)


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
        # TODO: remove this. This is only required for batch loading

        self._set_default_params(o) # TODO: maybe remove settings in opts.py

        #self.data, self.label = None, None
        self._load_data()
        self._split_data()

        self.ntr = self.data_tr['images'].shape[0]
        self.nte = self.data_te['images'].shape[0]
        self.idx_shuffle_tr = np.random.permutation(self.ntr)
        self.idx_shuffle_te = np.random.permutation(self.nte)

    def _set_default_params(self, o):
        # following paper's default parameter settings
        # TODO: remember some variables might need to set optional..

        self.frmsz = o.bouncing_mnist['frmsz']
        self.featdim = o.bouncing_mnist['featdim'] # TODO: change featdim (CNN)
        self.outdim = o.bouncing_mnist['outdim']

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
        '''
        mnist.h5 is already seperated in train and test! leave this empty.
        '''

    def _split_data(self):
        f = h5py.File(os.path.join(self.data_path, 'mnist.h5'))
        # f has 'train' and 'test' and each has 'inputs' and 'targets'
        self.data_tr = {'images': [], 'targets': []} 
        self.data_va = None # this data set doesn't seem to have separate val
        self.data_te = {'images': [], 'targets': []} 
        self.data_tr['images'] = np.asarray(f['train/inputs'].value)
        self.data_tr['targets'] = np.asarray(f['train/targets'].value)
        self.data_te['images'] = np.asarray(f['test/inputs'].value)
        self.data_te['targets'] = np.asarray(f['test/targets'].value)
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

    def run_sanitycheck(self, batch):
        draw.show_bouncing_mnist(batch)

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
                    data_='train') # TODO: this should be fine
        f = h5py.File(os.path.join(self.data_path,'ClutterPackLarge.hdf5', 'w'))
        f.create_dataset('clutterIMG', data=clutterIMG)
        f.close()
            
    def _GetFakeClutter(self):
        if self.clutterpack_exists:
            return self.clutterPack[np.random.randint(0, len(self.clutterPack))]
    
    def _GetClutter(self, image_size_ = None, num_clutters_ = None, fake = False, data_=None):
        if data_ == 'train':
            data_all = self.data_tr
        elif data_ == 'val':
            raise ValueError('val split not available for bouncing mnist (yet)')
        elif data_ == 'test':
            data_all = self.data_te
        else:
            raise ValueError('wrong data')

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

    #def GetBatch(self, o, verbose=False, count=1, data_=None):
    def get_batch(self, ib, o, verbose=False, count=1, data_=None):
        '''
        here in this function also made several changes in several places.
        '''
        if data_ == 'train':
            data_all = self.data_tr
            idx_shuffle = self.idx_shuffle_tr
        elif data_ == 'val':
            raise ValueError('val split not available for bouncing mnist (yet)')
        elif data_ == 'test':
            data_all = self.data_te
            idx_shuffle = self.idx_shuffle_te
        else:
            raise ValueError('wrong data')
        idx = idx_shuffle[(ib*o.batchsz):(ib+1)*o.batchsz] 
        
        start_y, start_x = self._GetRandomTrajectory(o.batchsz * self.num_digits_, o)
        window_y, window_x = self._GetRandomTrajectory(o.batchsz * 1, o, self.image_size_*2, object_size_=self.image_size_, step_length_ = 1e-2)
        # TODO: change data to real image or cluttered background
        data = np.zeros((o.batchsz, o.ntimesteps, self.image_size_, self.image_size_), dtype=np.float32)
        label = np.zeros((o.batchsz, o.ntimesteps, 4))
        for j in range(o.batchsz): 
            if np.random.random()<0.7 and self.buff and self.buff_cap > self.buff_size/2.0:
                data[j], label[j] = self._getBuff()
                continue
            else:
                clutter = self._GetClutter(fake=True, data_=data_)
                clutter_bg = self._GetClutter(fake=True, data_=data_)
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
                        label_offset = np.array([digit_image_nonzero[0].min(), digit_image_nonzero[1].min(), digit_image_nonzero[0].max(), digit_image_nonzero[1].max()])
 
                        wy=window_y[i, j]
                        wx=window_x[i, j]
                        data[j, i, top:bottom, left:right] = self._Overlap(data[j, i, top:bottom, left:right], scale_image)
                        data[j, i] = self._Overlap(data[j, i], clutter[wy:wy+self.image_size_, wx:wx+self.image_size_])
                        label[j, i] = label_offset + np.array([top, left, top, left])
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

        batch = {
                'inputs': data,
                'inputs_length': inputs_length, 
                'labels': label,
                'digits': data_all['targets'][idx],
                'idx': idx
                }

        return batch


def load_data(o):
    if o.dataset == 'moving_mnist':
        loader = Data_moving_mnist(o)
    elif o.dataset == 'bouncing_mnist':
        loader = Data_bouncing_mnist(o)
    else:
        raise ValueError('dataset not implemented yet')
    return loader

if __name__ == '__main__':
    print 'test dataset classes'

    # Test moving_mnist
    '''
    from opts import Opts
    o = Opts()
    o.batchsz = 20
    loader = load_data(o)
    batch = loader.get_batch(0, o, data_='train')
    '''

    # Test bouncing_mnist
    '''
    from opts import Opts
    o = Opts()
    o.batchsz = 20
    loader = Data_bouncing_mnist(o)
    batch = loader.get_batch(0, o, data_='train')
    #loader.run_sanitycheck(batch)
    '''

