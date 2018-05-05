'''Contains code for loading and installing datasets.

This module uses distinct concepts of datasets and subsets.
For example, the dataset 'ilsvrc' contains subsets 'ilsvrc_train' and 'ilsvrc_val'.
All image data for one dataset (e.g. 'ilsvrc') is contained in one dir (and put in one tarball).
Metadata is loaded for a subset using trackdat.load_xxx(dataset_dir, ...).

For speed, it is often useful to resize all images in a dataset.
To accommodate pre-processing, the data directory is arranged according to:
    {data_root}/{preproc}/{dataset}/
The un-pre-processed version is called "original".
Metadata is always loaded from the "original" version
in case the object is described in absolute pixel co-ordinates.
(This could be circumvented for ILSVRC since the image size is stored in the XML.
However, since the result is cached, it is a one-off cost.)
To load the metadata for a subset, use `load_metadata(data_root, subset_name)`.
The metadata includes the aspect ratio but not the size of the images.

Loading the metadata can be time consuming.
To avoid this, the metadata for each subset can be cached.

For running the code on a cluster where accessing individual files is slow,
it is possible to untar the dataset to a local drive.
This code can also be used to set up the data for the first time.
The module also include functionality to install the dataset from a tarball.
The function untar_and_load_all() installs and loads the metadata for several datasets.
'''

import msgpack
import numpy as np
import os
import subprocess
import time
from functools import partial
from itertools import chain

import logging
logger = logging.getLogger(__name__)

from seqtrack import helpers
import trackdat

CACHE_CODEC = msgpack
CACHE_EXT = '.msgpack'


class Subset(object):
    '''Describes a named subset of a dataset.

    The metadata of a subset is loaded using a load_xxx function in trackdat.
    For example, ilsvrc_train is a subset of ilsvrc, and is loaded using

        load_ilsvrc(dir, subset='train')
    '''

    def __init__(self, dataset, load_func):
        '''
        Args:
            dataset -- String.
        '''
        self.dataset = dataset
        self.load_func = load_func


SUBSETS = {
    'alov': Subset('alov', trackdat.load_alov),
    'dtb70': Subset('dtb70', trackdat.load_dtb70),
    'ilsvrc_train': Subset('ilsvrc', partial(trackdat.load_ilsvrc, subset='train')),
    'ilsvrc_val': Subset('ilsvrc', partial(trackdat.load_ilsvrc, subset='val')),
    'nfs_240': Subset('nfs', trackdat.load_nfs),
    'nfs_30': Subset('nfs', partial(trackdat.load_nfs, fps=30)),
    'nuspro': Subset('nuspro', trackdat.load_nuspro),
    'otb': Subset('otb', partial(trackdat.load_otb, subset='tb_100')),
    'otb_50': Subset('otb', partial(trackdat.load_otb, subset='tb_50')),
    'otb_cvpr13': Subset('otb', partial(trackdat.load_otb, subset='cvpr13')),
    'tc128': Subset('tc128', partial(trackdat.load_tc128, keep_prev=True)),
    'tc128_ce': Subset('tc128', partial(trackdat.load_tc128, keep_prev=False)),
    'tlp': Subset('tlp', trackdat.load_tlp),
    'uav123': Subset('uav123', partial(trackdat.load_uav123, subset='UAV123')),
    'uav20l': Subset('uav123', partial(trackdat.load_uav123, subset='UAV20L')),
    'vot2013': Subset('vot2013', trackdat.load_vot),
    'vot2014': Subset('vot2014', trackdat.load_vot),
    'vot2015': Subset('vot2015', trackdat.load_vot),
    'vot2016': Subset('vot2016', trackdat.load_vot),
    'vot2017': Subset('vot2017', trackdat.load_vot),
    'ytbb_train': Subset('ytbb', partial(trackdat.load_ytbb_sec, subset='train')),
    'ytbb_val': Subset('ytbb', partial(trackdat.load_ytbb_sec, subset='validation')),
}


def load(data_dir, preproc, subset_name, cache=False, cache_dir=None):
    '''Loads the metadata and instantiates.'''
    dataset_dir = os.path.join(data_dir, preproc, SUBSETS[subset_name].dataset)
    metadata = load_metadata(data_dir, subset_name, cache, cache_dir)
    return DatasetInstance(dataset_dir, metadata)


def load_metadata(data_dir, subset_name, cache=False, cache_dir=None):
    '''Loads metadata from {data_dir}/original/{subset}.

    Must be loaded from original data because
    metadata includes relative co-ordinates and aspect ratio,
    and pre-processed versions have different image sizes.
    '''
    if cache:
        return helpers.cache(trackdat.dataset.Serializer(CACHE_CODEC),
                             _cache_file(cache_dir, subset_name),
                             lambda: load_metadata(data_dir, subset_name))

    logger.info('load metadata: "%s"', subset_name)
    subset = SUBSETS[subset_name]
    start = time.time()
    metadata = subset.load_func(os.path.join(data_dir, 'original', subset.dataset))
    logger.info('time to load metadata: %.3g sec "%s"', time.time() - start, subset_name)
    return metadata


def untar_and_load_all(tar_dir, data_dir, preproc, subset_names, cache_dir):
    '''Untars and loads the metadata for many sets at once.

    If the metadata is not cached and preproc is not 'original',
    then the original dataset will also be untarred to load the metadata.

    These should be done together since multiple subsets can use the same dataset.
    The untar processes will also be executed in parallel.

    Expects directory structure:
        {tar_dir}/original/{dataset_name}.tar
        {tar_dir}/{preproc}/{dataset}.tar

    Creates directory structure:
        {data_dir}/{preproc}/{dataset}
        {cache_dir}/dataset_{subset_name}.json
    '''
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, 0755)

    if preproc != 'original':
        without_cache = [name for name in subset_names
                         if not os.path.isfile(_cache_file(cache_dir, name))]
        if len(without_cache) > 0:
            logger.info('cache not found for: %s', helpers.quote_list(without_cache))
            # Need to untar original version to load annotations.
            _untar(tar_dir, data_dir, 'original', _get_datasets(without_cache))
    else:
        _untar(tar_dir, data_dir, 'original', _get_datasets(subset_names))

    metadata = {name: load(data_dir, preproc, name, cache=True, cache_dir=cache_dir)
                for name in subset_names}

    if preproc != 'original':
        _untar(tar_dir, data_dir, preproc, _get_datasets(subset_names))
    return metadata


def _get_datasets(subset_names):
    return set(SUBSETS[name].dataset for name in subset_names)


def _cache_file(cache_dir, subset_name):
    return os.path.join(cache_dir, 'dataset_{}{}'.format(subset_name, CACHE_EXT))


def _untar(tar_dir, data_dir, preproc, names):
    '''
    Expects directory structure:
        {tar_dir}/{preproc}/{name}.tar
    Each tar file contains a single top-level directory {name}.

    Creates directory structure:
        {data_dir}/{preproc}/{name}/
    '''
    src_dir = os.path.join(tar_dir, preproc)
    tar_files = {name: os.path.join(src_dir, name + '.tar') for name in names}
    # First check if all tar files exist.
    for name in names:
        if not os.path.isfile(tar_files[name]):
            raise RuntimeError('tar file not found: "{}"'.format(tar_files[name]))
    dst_dir = os.path.join(data_dir, preproc)
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir, 0755)
    dataset_dirs = {name: os.path.join(dst_dir, name) for name in names}

    # Start a process to untar each file.
    start = time.time()
    procs = {}
    for name in names:
        logger.info('untar "%s" in "%s"', tar_files[name], dst_dir)
        procs[name] = subprocess.Popen(['tar', '-xf', tar_files[name]], cwd=dst_dir)
    while True:
        procs = {
            name: proc for name, proc in procs.items()
            if _still_running_or_assert_success(
                proc, name, tar_files[name], dataset_dirs[name], start)}
        if len(procs) == 0:
            break
        time.sleep(1)


def _still_running_or_assert_success(proc, name, tar_file, dataset_dir, start):
    retcode = proc.poll()
    if retcode is None:
        return True
    if retcode != 0:
        raise RuntimeError('could not untar "{}": non-zero return code {}'.format(
            tar_file, retcode))
    # Check that directory has been created.
    if not os.path.isdir(dataset_dir):
        raise RuntimeError('dir not created "{}"'.format(dataset_dir))
    logger.info('time to untar: %.0f sec "%s"', time.time() - start, dataset_dir)
    return False


class DatasetInstance(object):
    '''Describes dataset instantiated in a directory.

    Absolute image paths are returned.
    '''

    def __init__(self, dir, metadata):
        self.dir = dir
        self.metadata = metadata

    def tracks(self):
        return self.metadata.tracks()

    def video(self, track_id):
        return self.metadata.video(track_id)

    def labels(self, track_id):
        return self.metadata.labels(track_id)

    def image_file(self, video_id, time):
        '''Returns absolute path to image.'''
        return os.path.join(self.dir, self.metadata.image_file(video_id, time))

    def aspect(self, video_id):
        return self.metadata.aspect(video_id)


class Concat(object):
    '''Represents the concatenation of multiple datasets as one dataset.'''

    def __init__(self, datasets):
        '''
        Args:
            datasets: Dict that maps string to dataset.
        '''
        self.datasets = datasets

    def tracks(self):
        return [self.join_id(dataset_id, track_id)
                for dataset_id, dataset in self.datasets.items()
                for track_id in dataset.tracks()]

    def video(self, track_id):
        dataset_id, internal_track_id = self.split_id(track_id)
        internal_video_id = self.datasets[dataset_id].video(internal_track_id)
        return self.join_id(dataset_id, internal_video_id)

    def labels(self, track_id):
        dataset_id, internal_track_id = self.split_id(track_id)
        return self.datasets[dataset_id].labels(internal_track_id)

    def image_file(self, video_id, time):
        dataset_id, internal_id = self.split_id(video_id)
        return self.datasets[dataset_id].image_file(internal_id, time)

    def aspect(self, video_id):
        dataset_id, internal_id = self.split_id(video_id)
        return self.datasets[dataset_id].aspect(internal_id)

    def join_id(self, dataset_id, internal_id):
        assert '/' not in dataset_id
        return dataset_id + '/' + internal_id

    def split_id(self, external_id):
        dataset_id, internal_id = external_id.split('/', 1)
        return dataset_id, internal_id


class Subset(object):

    def __init__(self, dataset, track_subset):
        self.dataset = dataset
        self.track_subset = track_subset

    def tracks(self):
        return self.track_subset

    def video(self, track_id):
        return self.dataset.video(track_id)

    def labels(self, track_id):
        return self.dataset.labels(internal_track_id)

    def image_file(self, video_id, time):
        return self.dataset.image_file(video_id, time)

    def aspect(self, video_id):
        return self.dataset.aspect(video_id)


def get_videos(metadata):
    return sorted(set(map(metadata.video, metadata.tracks())))


def get_tracks_by_video(metadata):
    track_ids = {}
    for track_id in metadata.tracks():
        track_ids.setdefault(metadata.video(track_id), []).append(track_id)
    return track_ids


def split_dataset(dataset, pvals, seed=0):
    # Map from video_id to list of track_ids.
    tracks_by_video = get_tracks_by_video(dataset)
    videos = sorted(tracks_by_video.keys())
    rand = np.random.RandomState(seed)
    rand.shuffle(videos, seed)
    video_subsets = split_list(videos, pvals)
    assert sum(map(len, video_subsets)) == len(videos)
    track_subsets = [
        sorted(chain(*[tracks_by_video[video] for video in video_subsets[i]]))
        for i in range(len(video_subsets))]
    assert sum(map(len, track_subsets)) == len(dataset.tracks())
    return [Subset(dataset, track_subsets[i]) for i in range(len(track_subsets))]


def split_list(x, pvals):
    n = len(x)
    pvals = np.asfarray(pvals) / np.sum(pvals)
    stops = np.round(np.cumsum(pvals) * n).astype(np.int).tolist()
    starts = [0] + stops
    ys = []
    for start, stop in zip(starts, stops):
        ys.append(x[start:stop])
    return ys
