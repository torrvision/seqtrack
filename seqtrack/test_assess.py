from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

from seqtrack import geom_np
from seqtrack import assess


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
        subseqs = {'OPE': {name: [name] for name in seqs.keys()}}
        metrics = assess.assess_dataset(seqs, preds, subseqs)
        self.assertLessEqual(0, metrics['OPE_iou_seq_mean'])
        self.assertLessEqual(metrics['OPE_iou_seq_mean'], 1)
        for key in metrics:
            self.assertTrue(np.isscalar(metrics[key]),
                            'not scalar \'{}\': {}'.format(key, metrics[key]))


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
    names = list(map(str, range(num_seqs)))
    return dict(zip(names, seqs)), dict(zip(names, preds))
