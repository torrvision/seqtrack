from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import os

from seqtrack import geom_np
from seqtrack import assess

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(FILE_DIR, '..'))
OTB_DATA_DIR = os.path.join(PROJECT_DIR, 'aux', 'otb', 'test_data')
OTB_TRACKERS = [('KCF', 0.477), ('Staple', 0.581), ('CCOT', 0.671)]


class TestAssess(unittest.TestCase):

    def test_assess_frames(self):
        num_frames = 200
        seq, pred = random_sequence_and_predictions(np.random.RandomState(0), num_frames)
        metrics = assess.assess_frames(seq, pred)
        self.assertIn('iou', metrics.keys())
        self.assertEqual(len(metrics['iou']), num_frames - 1)
        assert all(0 <= iou <= 1 or not is_valid
                   for is_valid, iou in zip(seq['label_is_valid'][1:], metrics['iou']))

    def test_assess_sequence(self):
        num_frames = 200
        seq, pred = random_sequence_and_predictions(np.random.RandomState(0), num_frames)
        frame_metrics = assess.assess_frames(seq, pred)
        metrics = assess.assess_sequence(seq, pred, frame_metrics)
        self.assertIn('iou', metrics.keys())
        self.assertLessEqual(0, metrics['iou'])
        self.assertLessEqual(metrics['iou'], 1)
        self.assertLessEqual(metrics['iou'], metrics['oracle_size_iou'])

    def test_assess_dataset(self):
        num_seqs = 50
        seqs, preds = random_dataset_and_predictions(np.random.RandomState(0), num_seqs)
        metrics, _ = assess.assess_dataset(seqs, preds)
        self.assertLessEqual(0, metrics['OPE/iou_seq_mean'])
        self.assertLessEqual(metrics['OPE/iou_seq_mean'], 1)
        for key in metrics:
            self.assertTrue(np.isscalar(metrics[key]),
                            'not scalar \'{}\': {}'.format(key, metrics[key]))

    def test_assess_frames_vs_otb(self):
        '''Compares per-frame IOU against OTB benchmark.'''
        seqs = _load_otb_sequences()
        for tracker, _ in OTB_TRACKERS:
            preds, ious = _load_otb_predictions(tracker, list(seqs.keys()))
            for seq_name in seqs.keys():
                frame_metrics = assess.assess_frames(seqs[seq_name], preds[seq_name])
                np.testing.assert_allclose(frame_metrics['iou'], ious[seq_name], atol=1e-4)

    def test_assess_dataset_vs_otb(self):
        '''Compares area-under-curve against OTB benchmark.'''
        seqs = _load_otb_sequences()
        for tracker, desired_auc in OTB_TRACKERS:
            preds, ious = _load_otb_predictions(tracker, list(seqs.keys()))
            metrics, _ = assess.assess_dataset(seqs, preds)
            np.testing.assert_almost_equal(metrics['OPE/iou_success_auc_otb_seq_mean'],
                                           desired_auc, decimal=3)


def _load_otb_sequences():
    names = _read_lines(os.path.join(OTB_DATA_DIR, 'sequences.txt'))
    seqs = {}
    for name in names:
        fname = os.path.join(OTB_DATA_DIR, 'groundtruth', name + '.txt')
        rects = np.loadtxt(fname, delimiter=',')
        seq_len = len(rects)
        is_valid = np.all(rects > 0, axis=-1)
        rects = _convert_from_otb(rects)
        seqs[name] = {
            'image_files': [None for t in range(seq_len)],
            'labels': rects,
            'label_is_valid': is_valid,
            'aspect': [None for t in range(seq_len)],
            'video_name': name,
        }
    return seqs


def _convert_from_otb(rect, axis=-1):
    min_pt, size = np.split(rect, [2], axis=axis)
    return geom_np.make_rect(min_pt, min_pt + size)


def _load_otb_predictions(tracker, seq_names):
    preds = {}
    ious = {}
    for name in seq_names:
        fname = os.path.join(OTB_DATA_DIR, 'predictions', tracker, name + '.txt')
        data = np.loadtxt(fname, delimiter=',')
        seq_len = len(data)
        preds[name] = _convert_from_otb(data[1:, 0:4])
        iou = data[1:, 4]
        # OTB uses IOU = -1 for missing annotation.
        iou[iou < 0] = np.nan
        ious[name] = iou
    return preds, ious


def _read_lines(fname):
    with open(fname, 'r') as f:
        lines = [s.strip() for s in f.readlines()]
    return [s for s in lines if s]


def random_sequence_and_predictions(rand, num_frames, min_size=0.1, max_size=0.5,
                                    center_error=0.5, size_error=0.5, prob_valid=0.8):
    size = rand.uniform(min_size, max_size, size=(num_frames, 2))
    center = rand.uniform(max_size / 2, 1 - max_size / 2, size=(num_frames, 2))
    labels = geom_np.make_rect_center_size(center, size)
    # Perturb ground-truth to obtain prediction.
    pred_center = center + rand.uniform(-1, 1, size=(num_frames, 2)) * center_error * size
    pred_size = size + rand.uniform(-1, 1, size=(num_frames, 2)) * size_error * size
    preds = geom_np.make_rect_center_size(pred_center[1:], pred_size[1:])
    labels, is_valid = make_invalid(rand, labels, prob_valid=prob_valid)
    sequence = {'labels': labels, 'label_is_valid': is_valid}
    return sequence, preds


def make_invalid(rand, labels, prob_valid):
    num_frames, _ = labels.shape
    is_valid = rand.binomial(n=1, p=prob_valid, size=num_frames).astype(bool)
    is_valid[0] = True
    labels[np.logical_not(is_valid)] = float('nan')
    return labels, is_valid


def random_dataset_and_predictions(rand, num_seqs, min_len=200, max_len=400, **kwargs):
    lens = [int(round(x)) for x in rand.uniform(min_len, max_len, size=num_seqs)]
    seqs, preds = zip(*[random_sequence_and_predictions(rand, lens[i], **kwargs)
                        for i in range(num_seqs)])
    names = ['seq_{:d}'.format(i) for i in range(num_seqs)]
    return dict(zip(names, seqs)), dict(zip(names, preds))
