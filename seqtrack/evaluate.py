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
from seqtrack import helpers
from seqtrack import visualize as visualize_pkg
from seqtrack.helpers import load_image_viewport, im_to_arr, pad_to, to_nested_tuple
from seqtrack.helpers import escape_filename

FRAME_PATTERN = '%06d.jpeg'


def track(sess, model_inst, sequence, use_gt,
          verbose=False,
          # Visualization options:
          visualize=False,
          vis_dir=None,
          save_frames=False):
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

    if visualize:
        assert vis_dir is not None
        if not os.path.exists(vis_dir): os.makedirs(vis_dir, 0755)
        if not save_frames:
            frame_dir = tempfile.mkdtemp()
        else:
            frame_dir = os.path.join(vis_dir, 'frames', escape_filename(sequence['video_name']))
            if not os.path.exists(frame_dir): os.makedirs(frame_dir)

    # JV: Use viewport.
    # first_image = load_image(sequence['image_files'][0], model.image_size, resize=True)
    first_image = load_image_viewport(
        sequence['image_files'][0],
        sequence['viewports'][0],
        size_hw=(model_inst.imheight, model_inst.imwidth))
    first_label = sequence['labels'][0]
    # Prepare for input to network.
    batch_first_image = _single_to_batch(im_to_arr(first_image), model_inst.batchsz)
    batch_first_label = _single_to_batch(first_label, model_inst.batchsz)

    if visualize:
        im_vis = visualize_pkg.draw_output(first_image.copy(), rect_gt=first_label)
        im_vis.save(os.path.join(frame_dir, FRAME_PATTERN % 0))

    sequence_len = len(sequence['image_files'])
    assert(sequence_len >= 2)

    dur = 0.
    prev_time = time.time()

    # If the length of the sequence is greater than the instantiated RNN,
    # it will need to be run in chunks.
    y_pred_chunks = []
    # hmap_pred_chunks = []
    prev_state = {}
    for start in range(1, sequence_len, model_inst.ntimesteps):
        rem = sequence_len - start
        # Feed the next `chunk_len` frames into the model.
        chunk_len = min(rem, model_inst.ntimesteps)
        dur += time.time() - prev_time
        # JV: Use viewport.
        # images = map(lambda x: load_image(x, model.image_size, resize=True),
        #              sequence['image_files'][start:start+chunk_len]) # Create single array of all images.
        images = [
            load_image_viewport(image_file, viewport,
                                size_hw=(model_inst.imheight, model_inst.imwidth))
            for image_file, viewport in zip(
                sequence['image_files'][start:start + chunk_len],
                sequence['viewports'][start:start + chunk_len])
        ]
        prev_time = time.time()
        labels = sequence['labels'][start:start + chunk_len]
        is_valid = sequence['label_is_valid'][start:start + chunk_len]

        # Prepare data as input to network.
        images_arr = map(im_to_arr, images)
        feed_dict = {
            model_inst.example['x0']: batch_first_image,
            model_inst.example['y0']: batch_first_label,
            model_inst.example['x']:
                _single_to_batch(pad_to(images_arr, model_inst.ntimesteps), model_inst.batchsz),
            model_inst.example['y']:
                _single_to_batch(pad_to(labels, model_inst.ntimesteps), model_inst.batchsz),
            model_inst.example['y_is_valid']:
                _single_to_batch(pad_to(is_valid, model_inst.ntimesteps), model_inst.batchsz),
            model_inst.example['aspect']:
                _single_to_batch(sequence['aspect'], model_inst.batchsz),
            model_inst.run_opts['use_gt']: use_gt,
            model_inst.run_opts['is_tracking']: True,
        }
        if start > 1:
            # This is not the first chunk.
            # Add the previous state to the feed dictionary.
            tensor, value = to_nested_tuple(model_inst.state_init, prev_state)
            if tensor is not None:  # Function returns None if empty.
                feed_dict[tensor] = value
        # Get output and final state.
        # y_pred, prev_state, hmap_pred = sess.run(
        #     [model_inst.outputs['y']['ic'], model_inst.state_final, model_inst.outputs['hmap']['ic']],
        #     feed_dict=feed_dict)
        output_vars = {'y': model_inst.outputs['y']}
        if 'vis' in model_inst.outputs:
            output_vars['vis'] = model_inst.outputs['vis']
        outputs, prev_state = sess.run(
            [output_vars, model_inst.state_final], feed_dict=feed_dict)
        # Take first element of batch and first `chunk_len` elements of output.
        outputs['y'] = outputs['y'][0][:chunk_len]
        if 'vis' in outputs:
            outputs['vis'] = outputs['vis'][0][:chunk_len]

        if visualize:
            for i in range(len(images)):
                t = start + i
                if 'vis' in outputs:
                    image_i = Image.fromarray(np.uint8(255 * outputs['vis'][i]))
                else:
                    image_i = images[i]
                # im_vis = visualize_pkg.draw_output(images[i].copy(),
                #     rect_gt=(labels[i] if is_valid[i] else None),
                #     rect_pred=y_pred[i],
                #     hmap_pred=hmap_pred[i])
                im_vis = visualize_pkg.draw_output(image_i,
                                                   rect_gt=(labels[i] if is_valid[i] else None),
                                                   rect_pred=outputs['y'][i])
                im_vis.save(os.path.join(frame_dir, FRAME_PATTERN % t))

        y_pred_chunks.append(outputs['y'])
        # hmap_pred_chunks.append(hmap_pred)

    dur += time.time() - prev_time
    if verbose:
        print 'time: {:.3g} sec ({:.3g} fps)'.format(dur, (sequence_len - 1) / dur)

    if visualize:
        args = ['ffmpeg', '-loglevel', 'error',
                          # '-r', '1', # fps.
                          '-y',  # Overwrite without asking.
                          '-nostdin',  # No interaction with user.
                          '-i', FRAME_PATTERN,
                          '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                          os.path.join(os.path.abspath(vis_dir),
                                       escape_filename(sequence['video_name']) + '.mp4')]
        try:
            p = subprocess.Popen(args, cwd=frame_dir)
            p.wait()
        except Exception as inst:
            print 'error:', inst
        finally:
            if not save_frames:
                shutil.rmtree(frame_dir)

    # Concatenate the results for all chunks.
    y_pred = np.concatenate(y_pred_chunks)
    # hmap_pred = np.concatenate(hmap_pred_chunks)
    # return y_pred, hmap_pred
    return y_pred


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
    subseqs, tre_names = _split_tre_all(sequences, tre_num)
    predictions = {}
    bar = _make_progress_bar()
    for name in bar(subseqs.keys()):
        # TODO: Cache.
        # If we use a subset of sequences, we need to ensure that the subset is the same?
        # Could re-seed video sampler with global step number?
        predictions[name] = track(sess, model_inst, subseqs[name], use_gt=use_gt, **kwargs)
    return assess.assess_dataset(subseqs, predictions, tre_num, tre_names)


def _split_tre_all(sequences, tre_num):
    # sequences = list(sequences)
    subseqs = {}
    names = {}
    for sequence in sequences:
        seq_name = sequence['video_name']
        for tre_ind in range(tre_num):
            try:
                subseq = _extract_tre_sequence(sequence, tre_ind, tre_num)
            except RuntimeError as ex:
                logger.warning('sequence \'%s\': could not extract TRE sequence %d of %d: %s',
                               sequence['video_name'], tre_ind, tre_num, str(ex))
                continue
            subseq_name = subseq['video_name']
            if subseq_name in subseqs:
                raise RuntimeError('name already exists: \'{}\''.format(subseq_name))
            subseqs[subseq_name] = subseq
            names.setdefault(seq_name, []).append(subseq_name)
    return subseqs, names


def _tre_name(seq_name, i, n):
    if i == 0:
        return seq_name
    i, n = _simplify_fraction(i, n)
    return '{}_tre_{}_{}'.format(seq_name, i, n)


def _simplify_fraction(i, n):
    if i == 0:
        assert n != 0
        return 0, 1
    g = gcd(i, n)
    return i / g, n / g


def _extract_tre_sequence(seq, ind, num):
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
    subseq['video_name'] = _tre_name(seq['video_name'], ind, num)

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
