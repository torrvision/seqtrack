from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil
import subprocess
import tempfile
import time
from fractions import gcd
from PIL import Image

import logging
logger = logging.getLogger(__name__)

from seqtrack import assess
from seqtrack import data
from seqtrack import geom_np
from seqtrack import helpers
from seqtrack import visualize as visualize_pkg
from seqtrack.helpers import load_image_viewport, im_to_arr, pad_to, to_nested_tuple
from seqtrack.helpers import escape_filename

FRAME_PATTERN = '%06d.jpeg'


class ChunkedTracker(object):

    def __init__(self, sess, model_inst,
                 use_gt=False,
                 verbose=False,
                 sequence_name='untitled',
                 sequence_aspect=None,
                 # Visualization options:
                 visualize=False,
                 vis_dir=None,
                 keep_frames=False):
        self._sess = sess
        self._model_inst = model_inst
        self._use_gt = use_gt
        self._verbose = verbose
        self._sequence_name = sequence_name
        self._aspect = sequence_aspect
        self._visualize = visualize
        self._vis_dir = vis_dir
        self._keep_frames = keep_frames

        self._num_frames = 0  # Does not include initial frame.
        self._prev_state = {}
        self._start_time = 0
        self._duration_eval = 0  # Does not include initial frame.
        self._duration_with_load = 0  # Does not include initial frame.

        self._frame_dir = None
        if self._visualize:
            assert self._vis_dir is not None
            if not os.path.exists(self._vis_dir):
                os.makedirs(self._vis_dir, 0o755)
            if not self._keep_frames:
                self._frame_dir = tempfile.mkdtemp()
            else:
                self._frame_dir = os.path.join(self._vis_dir, 'frames', escape_filename(self._sequence_name))
                if not os.path.exists(self._frame_dir):
                    os.makedirs(self._frame_dir)

    def warmup(self):
        r = np.random.RandomState(0)
        first_image = np.random.normal(
            size=(self._model_inst.imheight, self._model_inst.imwidth, 3))
        first_label = geom_np.make_rect([0.4, 0.4], [0.6, 0.6])

        images_arr = np.random.normal(
            size=(self._model_inst.ntimesteps,
                  self._model_inst.imheight, self._model_inst.imwidth, 3))
        labels = [geom_np.make_rect([0.4, 0.4], [0.6, 0.6])
                  for _ in range(self._model_inst.ntimesteps)]
        is_valid = [True for _ in range(self._model_inst.ntimesteps)]
        aspect = 1

        feed_dict = {}
        feed_dict.update({
            self._model_inst.example['x0']: self._to_batch(first_image),
            self._model_inst.example['y0']: self._to_batch(first_label),
            self._model_inst.example['x']: self._to_batch_sequence(images_arr),
            self._model_inst.example['y']: self._to_batch_sequence(labels),
            self._model_inst.example['y_is_valid']: self._to_batch_sequence(is_valid),
            self._model_inst.example['aspect']: self._to_batch(aspect),
            self._model_inst.run_opts['use_gt']: self._use_gt,
            self._model_inst.run_opts['is_tracking']: True,
        })

        # Get output and final state.
        output_vars = {'y': self._model_inst.outputs['y'],
                       'score': self._model_inst.outputs['score']}
        if self._visualize and 'vis' in self._model_inst.outputs:
            output_vars['vis'] = self._model_inst.outputs['vis']
        for i in range(3):
            outputs, self._prev_state = self._sess.run(
                [output_vars, self._model_inst.state_final], feed_dict=feed_dict)


    def start(self, init_frame):
        '''
        Args:
            init_frame: Sequence of length 1.
        '''
        self._start_time = time.time()
        # JV: Use viewport.
        # first_image = load_image(sequence['image_files'][0], model.image_size, resize=True)
        if self._aspect is None:
            im_width, im_height = Image.open(init_frame['image_files'][0]).size
            self._aspect = float(im_width) / im_height
        first_image = load_image_viewport(
            init_frame['image_files'][0],
            init_frame['viewports'][0],
            size_hw=(self._model_inst.imheight, self._model_inst.imwidth))
        first_label = init_frame['labels'][0]
        # Prepare for input to network.
        self._batch_first_image = self._to_batch(im_to_arr(first_image))
        self._batch_first_label = self._to_batch(first_label)

        if self._visualize:
            im_vis = visualize_pkg.draw_output(first_image.copy(), rect_gt=first_label)
            im_vis.save(os.path.join(self._frame_dir, FRAME_PATTERN % 0))

    def next(self, chunk):
        chunk_len = len(chunk['image_files'])
        assert chunk_len <= self._model_inst.ntimesteps
        # TODO: If chunk_len != self._model.sequence_len, then set self._final.

        feed_dict = {}
        if self._num_frames > 0:
            # This is not the first chunk.
            # Add the previous state to the feed dictionary.
            tensor, value = to_nested_tuple(self._model_inst.state_init, self._prev_state)
            if tensor is not None:  # Function returns None if empty.
                feed_dict[tensor] = value

        start_load = time.time()
        image_size_hw = (self._model_inst.imheight, self._model_inst.imwidth)
        images = [
            load_image_viewport(image_file, viewport, image_size_hw)
            for image_file, viewport in zip(chunk['image_files'], chunk['viewports'])
        ]
        labels = chunk['labels']
        is_valid = chunk['label_is_valid']

        # Prepare data as input to network.
        images_arr = list(map(im_to_arr, images))
        feed_dict.update({
            self._model_inst.example['x0']: self._batch_first_image,
            self._model_inst.example['y0']: self._batch_first_label,
            self._model_inst.example['x']: self._to_batch_sequence(images_arr),
            self._model_inst.example['y']: self._to_batch_sequence(labels),
            self._model_inst.example['y_is_valid']: self._to_batch_sequence(is_valid),
            self._model_inst.example['aspect']: self._to_batch(self._aspect),
            self._model_inst.run_opts['use_gt']: self._use_gt,
            self._model_inst.run_opts['is_tracking']: True,
        })

        # Get output and final state.
        output_vars = {'y': self._model_inst.outputs['y'],
                       'score': self._model_inst.outputs['score']}
        if self._visualize and 'vis' in self._model_inst.outputs:
            output_vars['vis'] = self._model_inst.outputs['vis']
        start_run = time.time()
        outputs, self._prev_state = self._sess.run(
            [output_vars, self._model_inst.state_final], feed_dict=feed_dict)

        self._duration_eval += time.time() - start_run
        self._duration_with_load += time.time() - start_load

        # Take first element of batch and first `chunk_len` elements of output.
        outputs['y'] = outputs['y'][0][:chunk_len]
        outputs['score'] = outputs['score'][0][:chunk_len]
        if self._visualize and 'vis' in self._model_inst.outputs:
            outputs['vis'] = outputs['vis'][0][:chunk_len]

        # if self._visualize:
        #     for i in range(len(images)):
        #         t = self._num_frames + i + 1
        #         im_vis = visualize_pkg.draw_output(
        #             images[i].copy(),
        #             rect_gt=(labels[i] if is_valid[i] else None),
        #             rect_pred=y_pred[i],
        #             hmap_pred=hmap_pred[i])
        #         im_vis.save(os.path.join(self._frame_dir, FRAME_PATTERN % t))

        self._num_frames += chunk_len
        # return y_pred, hmap_pred
        return outputs

    def end(self):
        end_time = time.time()

        if self._visualize:
            args = [
                'ffmpeg',
                '-loglevel', 'error',
                # '-r', '1', # fps.
                '-y',  # Overwrite without asking.
                '-nostdin',  # No interaction with user.
                '-i', FRAME_PATTERN,
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                os.path.join(os.path.abspath(self._vis_dir),
                             escape_filename(self._sequence_name) + '.mp4')]
            try:
                subprocess.check_call(args, cwd=self._frame_dir)
            except Exception as ex:
                logger.warning('error calling ffmpeg: %s', str(ex))
            finally:
                if not self._keep_frames:
                    shutil.rmtree(self._frame_dir)

        timing = {}
        timing['duration_eval'] = self._duration_eval
        timing['duration_with_load'] = self._duration_with_load
        timing['duration_real'] = end_time - self._start_time
        timing['num_frames'] = self._num_frames
        return timing

    def _to_batch(self, x):
        return _single_to_batch(x, self._model_inst.batchsz)

    def _to_batch_sequence(self, x):
        return _single_to_batch(pad_to(x, self._model_inst.ntimesteps), self._model_inst.batchsz)


def track(sess, model_inst, sequence, use_gt,
          verbose=False,
          # Visualization options:
          visualize=False,
          vis_dir=None,
          keep_frames=False):
    '''Run an instantiated tracker on a sequence.

    model_inst.batchsz      -- Integer or None
    model_inst.ntimesteps   -- Integer
    model_inst.imheight     -- Integer
    model_inst.imwidth      -- Integer
    model_inst.example      -- Dictionary of tensors
    model_inst.run_opts     -- Dictionary of tensors
    model_inst.outputs      -- Dictionary of tensors
    model_inst.state_init   -- Nested collection of tensors.
    model_inst.state_final  -- Nested collection of tensors.

    sequence['image_files']    -- List of strings of length n.
    sequence['viewports']      -- Numpy array of rectangles [n, 4].
    sequence['labels']         -- Numpy array of shape [n, 4]
    sequence['label_is_valid'] -- List of booleans of length n.
    sequence['aspect']         -- Aspect ratio of original image.
        Required to compute IOU, etc. with correct aspect ratio.
    '''
    # TODO: Variable batch size.
    # TODO: Run on a batch of sequences for speed.

    tracker = ChunkedTracker(
        sess, model_inst,
        use_gt=use_gt,
        verbose=verbose,
        sequence_name=sequence['video_name'],
        sequence_aspect=sequence['aspect'],
        visualize=visualize,
        vis_dir=vis_dir,
        keep_frames=keep_frames,
    )
    init_frame = {
        'image_files': sequence['image_files'][0:1],
        'viewports': sequence['viewports'][0:1],
        'labels': sequence['labels'][0:1],
        'is_valid': sequence['label_is_valid'][0:1],
    }
    tracker.start(init_frame)

    sequence_len = len(sequence['image_files'])
    assert(sequence_len >= 2)

    # If the length of the sequence is greater than the instantiated RNN,
    # it will need to be run in chunks.
    output_chunks = []
    for start in range(1, sequence_len, model_inst.ntimesteps):
        rem = sequence_len - start
        # Feed the next `chunk_len` frames into the model.
        chunk_len = min(rem, model_inst.ntimesteps)

        input_chunk = {
            'image_files': sequence['image_files'][start:start + chunk_len],
            'viewports': sequence['viewports'][start:start + chunk_len],
            'labels': sequence['labels'][start:start + chunk_len],
            'label_is_valid': sequence['label_is_valid'][start:start + chunk_len],
        }
        output_chunk = tracker.next(input_chunk)
        output_chunks.append(output_chunk)

    info = tracker.end()

    # Concatenate the results for all chunks.
    # y_pred = np.concatenate(y_pred_chunks)
    # hmap_pred = np.concatenate(hmap_pred_chunks)
    y_pred = np.concatenate([chunk['y'] for chunk in output_chunks])
    return y_pred, info


class SimpleTracker(object):
    '''Describes a frame-by-frame tracker.'''

    def __init__(self, sess, model_inst, **kwargs):
        '''
        kwargs for ChunkedTracker
        '''
        self._tracker = ChunkedTracker(sess, model_inst, **kwargs)

    def warmup(self):
        self._tracker.warmup()

    def start(self, image_file, rect):
        init_frame = {
            'image_files': [image_file],
            'viewports': [geom_np.unit_rect()],
            'labels': [rect],
        }
        self._tracker.start(init_frame)

    def next(self, image_file, gt_rect=None):
        label_valid = gt_rect is not None
        chunk = {
            'image_files': [image_file],
            'viewports': [geom_np.unit_rect()],
            'labels': [gt_rect if label_valid else geom_np.unit_rect()],
            'label_is_valid': [label_valid],
        }
        outputs = self._tracker.next(chunk)
        # Extract single frame from sequence.
        outputs = {k: _only(v) for k, v in outputs.items()}
        return outputs

    def end(self):
        return self._tracker.end()


def _only(xs):
    x, = xs
    return x


def _single_to_batch(x, batch_size):
    x = np.expand_dims(x, 0)
    if batch_size is None:
        return x
    return pad_to(x, batch_size)


def _make_progress_bar():
    # return progressbar.ProgressBar(widgets=[
    #     progressbar.SimpleProgress(format='sequence %(value_s)s/%(max_value_s)s'), ' ',
    #     '(', progressbar.Percentage(), ') ',
    #     progressbar.Bar(), ' ',
    #     progressbar.Timer(), ' (', progressbar.ETA(format_finished='ETA: Complete'), ')',
    # ])
    return helpers.ProgressMeter(interval_time=60)


def evaluate_model(sess, model_inst, sequences, use_gt=False, tre_num=1, **kwargs):
    '''
    Returns:
        A dictionary that contains evaluation results.
    '''
    subseqs, tre_groups = _split_tre_all(sequences, tre_num)
    predictions = {}
    timing = {}
    bar = _make_progress_bar()
    for name in bar(subseqs.keys()):
        # TODO: Cache.
        # If we use a subset of sequences, we need to ensure that the subset is the same?
        # Could re-seed video sampler with global step number?
        predictions[name], timing[name] = track(
            sess, model_inst, subseqs[name], use_gt=use_gt, **kwargs)
    return assess.assess_dataset(subseqs, predictions, tre_groups, timing=timing)


def _split_tre_all(sequences, tre_num):
    sequences = list(sequences)

    subseqs = {}
    for sequence in sequences:
        for tre_ind in range(tre_num):
            # subseq_name = subseq['video_name']
            subseq_name = _tre_seq_name(sequence['video_name'], tre_ind, tre_num)
            if subseq_name in subseqs:
                raise RuntimeError('name already exists: \'{}\''.format(subseq_name))
            try:
                subseq = _extract_tre_sequence(sequence, tre_ind, tre_num, subseq_name)
            except RuntimeError as ex:
                logger.warning('sequence \'%s\': could not extract TRE sequence %d of %d: %s',
                               sequence['video_name'], tre_ind, tre_num, str(ex))
                # Still insert None if sequence could not be extracted.
                subseq = None
            subseqs[subseq_name] = subseq

    # Build sequence groups for OPE and TRE mode.
    # TODO: Could add all integer factors here?
    tre_group_nums = set([1, tre_num])  # If tre_num is 1, then only OPE.
    tre_groups = {}
    for tre_group_num in tre_group_nums:
        mode = _mode_name(tre_group_num)
        tre_groups[mode] = {}
        for sequence in sequences:
            seq_name = sequence['video_name']
            tre_groups[mode][seq_name] = []
            for tre_ind in range(tre_group_num):
                subseq_name = _tre_seq_name(seq_name, tre_ind, tre_group_num)
                if subseqs[subseq_name] is not None:
                    tre_groups[mode][seq_name].append(subseq_name)
    return subseqs, tre_groups


def _mode_name(tre_num):
    if tre_num == 1:
        return 'OPE'
    elif tre_num > 1:
        return 'TRE_{}'.format(tre_num)
    else:
        raise RuntimeError('tre_num must be at least one: {}'.format(tre_num))


def _tre_seq_name(seq_name, i, n):
    if i == 0:
        return seq_name
    i, n = _simplify_fraction(i, n)
    return '{}_tre_{}_{}'.format(seq_name, i, n)


def _simplify_fraction(i, n):
    if i == 0:
        assert n != 0
        return 0, 1
    g = gcd(i, n)
    return i // g, n // g


def _extract_tre_sequence(seq, ind, num, subseq_name):
    if ind == 0:
        return seq

    is_valid = seq['label_is_valid']
    num_frames = len(is_valid)
    valid_frames = [t for t in range(num_frames) if is_valid[t]]
    min_t, max_t = valid_frames[0], valid_frames[-1]

    start_t = int(round(float(ind) / num * (max_t - min_t))) + min_t
    # Snap to next valid frame (raises StopIteration if none).
    try:
        start_t = next(t for t in valid_frames if t >= start_t)
    except StopIteration:
        raise RuntimeError('no frames with labels')

    subseq = _extract_interval(seq, start_t, None)
    # subseq['video_name'] = seq['video_name'] + ('-seg{}'.format(ind) if ind > 0 else '')
    subseq['video_name'] = subseq_name

    # if len(subseq['image_files']) < 2:
    #     raise RuntimeError('sequence shorter than two frames')
    # Count number of valid labels after first frame (ground truth).
    num_valid = sum(1 for x in subseq['label_is_valid'][1:] if x)
    if num_valid < 1:
        raise RuntimeError('no frames with labels after first frame')
    return subseq


def _extract_interval(seq, start, stop):
    KEYS_SEQ = ['image_files', 'viewports', 'labels', 'label_is_valid']
    KEYS_NON_SEQ = ['aspect']
    subseq = {}
    subseq.update({k: seq[k][start:stop] for k in KEYS_SEQ})
    subseq.update({k: seq[k] for k in KEYS_NON_SEQ})
    return subseq
