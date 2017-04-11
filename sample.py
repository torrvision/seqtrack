'''The functions in this file create collections of sequences from datasets.

The sequences can be returned as a list or the functions can act as generators.
'''

import pdb
import numpy as np
import os
import random


#def sample(dataset, ntimesteps, seqtype=None, shuffle=True):
#    num_videos = len(dataset.videos)
#    if shuffle:
#        idx_shuffle = np.random.permutation(num_videos)
#    else:
#        idx_shuffle = range(num_videos)
#
#    def _select_frms(objvalidfrms):
#        if seqtype == 'dense':
#            '''
#            Select one sequence only from (possibly multiple) consecutive ones.
#            All frames are annotated (dense).
#            eg, [1,1,1,1,1,1,1,1,1]
#            '''
#            # firstly create consecutive 1s
#            segment_minlen = 2
#            consecutiveones = []
#            stack = []
#            for i, val in enumerate(objvalidfrms):
#                if val == 0:
#                    if len(stack) >= segment_minlen:
#                        consecutiveones.append(stack)
#                    stack = []
#                elif val == 1:
#                    stack.append(i)
#                else:
#                    raise ValueError('should be either 1 or 0')
#            if len(stack) >= segment_minlen: consecutiveones.append(stack)
#
#            # randomly choose one segment
#            frms_cand = random.choice(consecutiveones)
#
#            # select frames (randomness in it and < RNN+1 size)
#            frm_length = np.minimum(
#                random.randint(segment_minlen, len(frms_cand)), ntimesteps+1)
#            frm_start = random.randint(0, len(frms_cand)-frm_length)
#            frms = frms_cand[frm_start:frm_start+frm_length]
#        elif seqtype == 'sparse':
#            '''
#            Select from first valid frame to last valid frame.
#            It is possible that some frames are missing labels (sparse).
#            eg, [1,1,1,0,0,1,0,1,1]
#            '''
#            objvalidfrms_np = np.asarray(objvalidfrms)
#            assert(objvalidfrms_np.ndim==1)
#            # first one and last one
#            frms_one = np.where(objvalidfrms_np == 1)
#            frm_start = frms_one[0][0]
#            frm_end = frms_one[0][-1]
#            frms = range(frm_start, frm_end+1)
#        elif seqtype == 'sampling':
#            '''
#            Sampling k number of 1s from first 1 to last 1.
#            It consists of only 1s (no sparse), but is sampled from long range.
#            Example outputs can be [1,1,1,1,1,1].
#            '''
#            # 1. get selection range (= first one and last one)
#            objvalidfrms_np = np.asarray(objvalidfrms)
#            frms_one = np.where(objvalidfrms_np == 1)[0]
#            frm_start = frms_one[0]
#            frm_end = frms_one[-1]
#            # 2. get possible min and max (max <= o.ntimesteps+1; min >= 2)
#            minlen = 2
#            maxlen = min(frm_end-frm_start+1, ntimesteps+1)
#            # 3. decide length k (sampling number; min <= k <= max)
#            #k = np.random.randint(minlen, maxlen+1, 1)[0]
#            # NOTE: if k is in [min,max], it results in too many <T+1 sequences.
#            # But I think I need a lot more samples that is of length T+1.
#            # So, I stick to have k=T+1 if possible, but change later if necessary.
#            k = maxlen
#            # 4. sample k uniformly, using randperm
#            indices = np.random.randint(0, frms_one.size, k)
#            # 5. find frames
#            frms = frms_one[np.sort(indices)].tolist()
#        else:
#            raise ValueError('not available option for seqtype.')
#        return frms
#
#    for idx in idx_shuffle:
#        video = dataset.videos[idx]
#        # NOTE: examples have a length of ntimesteps+1.
#        files = []
#        labels = []
#        # randomly select an object
#        trajectory = random.choice(dataset.tracks[video])
#        if len(trajectory) < 2:
#            continue
#        # randomly select segment of frames (<=T+1)
#        # TODO: if seqtype is not dense any more, should change 'inputs_valid' in the below as well.
#        # self.objvalidfrms_snp should be changed as well.
#        frm_is_valid = [(t in trajectory) for t in range(dataset.video_length[video])]
#        pdb.set_trace()
#        frames = _select_frms(frm_is_valid)
#        yield {
#            'image_files':    [dataset.image_file(video, t) for t in frames],
#            'labels':         [trajectory.get(t, _invalid_rect()) for t in frames],
#            'label_is_valid': [(t in trajectory) for t in frames],
#            'original_image_size': dataset.original_image_size[video],
#        }
#
#
#def all_tracks_full(dataset, seqtype=None):
#    def _select_frms(objvalidfrms):
#        # 1. get selection range (= first one and last one)
#        objvalidfrms_np = np.asarray(objvalidfrms)
#        frms_one = np.where(objvalidfrms_np == 1)[0]
#        frm_start = frms_one[0]
#        frm_end = frms_one[-1]
#        return range(frm_start, frm_end+1)
#
#    for video in dataset.videos:
#        files = []
#        labels = []
#        for trajectory in dataset.tracks[video]:
#            if len(trajectory) < 2:
#                continue
#            frm_is_valid = [(t in trajectory) for t in range(dataset.video_length[video])]
#            frames = _select_frms(frm_is_valid)
#            yield {
#                'image_files':    [dataset.image_file(video, t) for t in frames],
#                'labels':         [trajectory.get(t, _invalid_rect()) for t in frames],
#                'label_is_valid': [(t in trajectory) for t in frames],
#                'original_image_size': dataset.original_image_size[video],
#            }
#
#
#def _invalid_rect():
#    return [float('nan')] * 4

def sample(dataset, ntimesteps=None, seqtype=None, shuffle=False):
    def _select_frames(frame_is_valid, frames_with_label):
        if seqtype == 'sampling':
            ''' Sample frames with `freq`, regardless of label
            (only the first frame need to have label).
            Thus, the returned frames can be `SPARSE`, e.g., [1,1,1,0,1,0,0].
            Note also that the returned frames can have length < ntimesteps+1.
            Adaptive frequency or gradually increasing frequency as a
            Curriculum Learning might be tried.
            '''
            freq = 10
            sampled_frames = range(random.choice(frames_with_label), len(frame_is_valid), freq)
            return sampled_frames[:ntimesteps+1]
        elif seqtype == 'full':
            ''' The full sequence from first 1 to last 1, regardless of label.
            Thus, the returned frames can be `SPARSE`, e.g., [1,1,1,1,0,0,1,1].
            This option is used to evaluate full-length sequences.
            '''
            return range(frames_with_label[0], frames_with_label[-1])

    num_videos = len(dataset.videos)
    indices = np.random.permutation(num_videos) if shuffle else range(num_videos)
    videos = list(dataset.videos[i] for i in indices)
    for video in videos:
        if ntimesteps and seqtype is not 'full':
            trajectories = [random.choice(dataset.tracks[video])]
        else:
            trajectories = dataset.tracks[video]
        for trajectory in trajectories:
            frame_is_valid = [(t in trajectory) for t in range(dataset.video_length[video])]
            frames = _select_frames(frame_is_valid, trajectory.keys())
            yield {
                'image_files':    [dataset.image_file(video, t) for t in frames],
                'labels':         [trajectory.get(t, _invalid_rect()) for t in frames],
                'label_is_valid': [(t in trajectory) for t in frames],
                'original_image_size': dataset.original_image_size[video],
            }

def _invalid_rect():
    return [float('nan')] * 4
