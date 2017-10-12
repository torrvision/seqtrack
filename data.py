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

import csv
import glob
import itertools
import os
import random
import xmltodict
import cv2
from PIL import Image

import draw
import helpers


class Data_ILSVRC(object):
    def __init__(self, dstype, o):
        self.dstype      = dstype
        self.path_data   = os.path.join(o.path_data_home, 'ILSVRC')
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
        ## self._update_stat(o)

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

    ## def _update_stat(self, o):
    ##     def create_stat_pixelwise(): # NOTE: obsolete
    ##         stat = dict.fromkeys({'mean', 'std'}, None)
    ##         mean = []
    ##         std = []
    ##         for i, video in enumerate(self.videos):
    ##             print 'computing mean and std in snippet of {}, {}/{}'.format(
    ##                     self.dstype, i+1, len(self.videos))
    ##             imglist = sorted(glob.glob(os.path.join(self._images_dir(video), '*.JPEG')))
    ##             xs = []
    ##             for j in imglist:
    ##                 # NOTE: perform resize image!
    ##                 xs.append(cv2.resize(cv2.imread(j)[:,:,(2,1,0)], 
    ##                     (o.frmsz, o.frmsz), interpolation=cv2.INTER_AREA))
    ##             xs = np.asarray(xs)
    ##             mean.append(np.mean(xs, axis=0))
    ##             std.append(np.std(xs, axis=0))
    ##         mean = np.mean(np.asarray(mean), axis=0)
    ##         std = np.mean(np.asarray(std), axis=0)
    ##         stat['mean'] = mean
    ##         stat['std'] = std
    ##         return stat

    ##     def create_stat_global():
    ##         stat = dict.fromkeys({'mean', 'std'}, None)
    ##         means = []
    ##         stds = []
    ##         for i, video in enumerate(self.videos):
    ##             print 'computing mean and std in snippet of {}, {}/{}'.format(
    ##                     self.dstype, i+1, len(self.videos))
    ##             imglist = sorted(glob.glob(os.path.join(self._images_dir(video), '*.JPEG')))
    ##             xs = []
    ##             # for j in imglist:
    ##             #     # NOTE: perform resize image!
    ##             #     xs.append(cv2.resize(cv2.imread(j)[:,:,(2,1,0)], 
    ##             #         (o.frmsz, o.frmsz), interpolation=cv2.INTER_AREA))
    ##             # Lazy method: Read a single image, do not resize.
    ##             xs.append(cv2.imread(imglist[len(imglist)/2])[:,:,(2,1,0)])
    ##             means.append(np.mean(xs))
    ##             stds.append(np.std(xs))
    ##         mean = np.mean(means)
    ##         std = np.mean(stds)
    ##         stat['mean'] = mean
    ##         stat['std'] = std
    ##         return stat

    ##     if self.stat is None:
    ##         if self.dstype == 'train':
    ##             filename = os.path.join(o.path_stat, 
    ##                 'meanstd_{}_frmsz_{}_train_{}.npy'.format(o.dataset, o.frmsz, self.trainsplit))
    ##         else:
    ##             filename = os.path.join(o.path_stat,
    ##                 'meanstd_{}_frmsz_{}_{}.npy'.format(o.dataset, o.frmsz, self.dstype))
    ##         if os.path.exists(filename):
    ##             self.stat = np.load(filename).tolist()
    ##         else:
    ##             self.stat = create_stat_global() 
    ##             np.save(filename, self.stat)


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
                if video == 'Human4' and gt_file == 'groundtruth_rect.1.txt':
                    continue
                rects = load_rectangles(os.path.join(video_dir, gt_file))
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
            assert(len(rects) > 0)
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


class Concat:

    def __init__(self, datasets):
        '''
        Args:
            datasets: Dictionary that maps name (string) to dataset.
        '''
        assert all('/' not in name for name in datasets.keys())
        assert all(' ' not in name for name in datasets.keys())
        self.datasets = datasets
        # Copy all fields.
        self.videos              = []
        self.video_length        = {}
        # self.image_size          = {}
        self.original_image_size = {}
        self.tracks              = {}
        for dataset_name, dataset in datasets.items():
            for video in dataset.videos:
                new_video = dataset_name + '/' + video
                self.videos.append(new_video)
                self.video_length[new_video]        = dataset.video_length[video]
                # self.image_size[new_video]          = dataset.image_size[video]
                self.original_image_size[new_video] = dataset.original_image_size[video]
                self.tracks[new_video]              = dataset.tracks[video]

    def image_file(self, video, frame):
        dataset_name, old_video = video.split('/', 1)
        return self.datasets[dataset_name].image_file(old_video, frame)


class CSV:

    def __init__(self, name, o):
        # Video name is directory of image file.
        fname = os.path.join(o.path_data_home, 'csv', name+'.csv')
        with open(fname, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            rects = {}
            size = {}
            format_str = None
            for row in reader:
                video_path, image = os.path.split(row['filename'])
                dataset_path, video_dir = os.path.split(video_path)
                track_id = row['trackID']
                image_name, image_ext = os.path.splitext(image)
                t = int(image_name)
                if not format_str:
                    # e.g. '{:06d}.JPEG'
                    # It is necessary to determine the format string to find
                    # the path of unlabelled frames, which do not appear in the csv file.
                    format_str = '{:0' + str(len(image_name)) + '}' + image_ext
                assert image == format_str.format(t)
                rects.setdefault(video_dir, {}).setdefault(track_id, {})[t] = CSV._row_to_rect(row)
                size[video_dir] = (int(row['frameWidth']), int(row['frameHeight']))

        videos = sorted(rects.keys())
        # Convert dictionaries of tracks to lists.
        rects = {
            video: [tracks[name] for name in sorted(tracks.keys())]
            for video, tracks in rects.iteritems()
        }
        # Take maximum time + 1 as video length.
        video_length = {
            video: max(t for track in tracks for t in track.keys())+1
            for video, tracks in rects.iteritems()
        }

        self.videos = videos
        self.video_length = video_length
        self.tracks = rects
        self.original_image_size = size
        self.format_str = format_str
        self.dataset_path = dataset_path
        if o.useresizedimg:
            self.image_dir = os.path.join(o.path_data_home, 'csv', 'images_frmsz{}'.format(o.frmsz))
        else:
            # TODO: Use something different here?
            self.image_dir = os.path.join(o.path_data_home, 'csv', 'images')

    @staticmethod
    def _row_to_rect(row):
        min_x  = float(row['leftX'])  / float(row['frameWidth'])
        min_y  = float(row['topY'])   / float(row['frameHeight'])
        size_x = float(row['width'])  / float(row['frameWidth'])
        size_y = float(row['height']) / float(row['frameHeight'])
        return [min_x, min_y, min_x+size_x, min_y + size_y]

    def image_file(self, video, frame):
        return os.path.join(self.image_dir, self.dataset_path, video, self.format_str.format(frame))


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

