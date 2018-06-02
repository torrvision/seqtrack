from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import json
import math
import numpy as np
import os
import pprint
import tempfile

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import cnnutil
from seqtrack import helpers
from seqtrack import search
from seqtrack import slurm
from seqtrack import train
from seqtrack.models import util


def _train(args, name, vector):
    pprint.pprint(vector)

    # Split vector.
    _assert_partition(set(vector.keys()), [KEYS_TRAIN, KEYS_SIAMFC])
    train_vector = {k: vector[k] for k in KEYS_TRAIN}
    model_vector = {k: vector[k] for k in KEYS_SIAMFC}

    train_kwargs, train_info = train_vector_to_kwargs(train_vector)
    model_kwargs, model_info = siamfc_vector_to_kwargs(model_vector)
    # Merge info dicts.
    info = dict(itertools.chain(train_info.items(), model_info.items()))

    tmp_dir = _get_tmp_dir()
    metrics = train.train(
        dir=os.path.join('trials', name), model_params=model_kwargs, seed=0,
        summary_dir='summary', summary_name=name,
        verbose_train=args.verbose_train,
        # Args from app.add_tracker_config_args()
        use_queues=args.use_queues,
        nosave=args.nosave,
        period_ckpt=args.period_ckpt,
        period_assess=args.period_assess,
        period_skip=args.period_skip,
        period_summary=args.period_summary,
        period_preview=args.period_preview,
        # save_videos=args.save_videos,
        save_frames=args.save_frames,
        session_config_kwargs=dict(
            gpu_manctrl=args.gpu_manctrl, gpu_frac=args.gpu_frac,
            log_device_placement=args.log_device_placement),
        # Arguments required for setting up data.
        # TODO: How to make this a parameter?
        train_dataset=args.train_dataset,
        val_dataset=args.val_dataset,
        eval_datasets=args.eval_datasets,
        pool_datasets=args.pool_datasets,
        pool_split=args.pool_split,
        untar=args.untar,
        data_dir=args.data_dir,
        tar_dir=args.tar_dir,
        tmp_data_dir=os.path.join(tmp_dir, 'data'),
        preproc_id=args.preproc,
        data_cache_dir=args.data_cache_dir,
        # Sampling:
        augment_motion=False,
        motion_params=None,
        # Args from app.add_eval_args()
        eval_tre_num=args.eval_tre_num,
        eval_samplers=args.eval_samplers,
        max_eval_videos=args.max_eval_videos,
        # Training process:
        imwidth=args.imwidth,
        imheight=args.imheight,
        # ntimesteps=None,
        # batchsz=None,
        # imwidth=None,
        # imheight=None,
        # lr_init=None,
        # lr_decay_steps=None,
        # lr_decay_rate=None,
        # optimizer=None,
        # weight_decay=None,
        # grad_clip=None,
        # max_grad_norm=None,
        # siamese_pretrain=None,
        # siamese_model_file=None,
        num_steps=args.num_steps,
        use_gt_train=False,
        gt_decay_rate=None,
        min_gt_ratio=None,
        # The hyper-parameters from the vector.
        **train_kwargs)

    return metrics


def _get_tmp_dir():
    if _is_slurm_job():
        return '/raid/local_scratch/{}-{}'.format(
            os.environ['SLURM_JOB_USER'], os.environ['SLURM_JOB_ID'])
    else:
        return tempfile.mkdtemp()


def _is_slurm_job():
    return 'SLURM_JOB_ID' in os.environ


def _assert_partition(x, subsets):
    assert isinstance(x, set)
    counts = {}
    for subset in subsets:
        for elem in subset:
            counts[elem] = counts.get(elem, 0) + 1
    union = set(counts.keys())
    missing = x.difference(union)
    if len(missing) > 0:
        raise RuntimeError('missing from partition: {}'.format(helpers.quote_list(missing)))
    extra = union.difference(x)
    if len(extra) > 0:
        raise RuntimeError('extra in partition: {}'.format(helpers.quote_list(extra)))
    multiple = [elem for elem, count in counts.items() if count > 1]
    if len(multiple) > 0:
        raise RuntimeError('repeated in partition: {}'.format(helpers.quote_list(multiple)))
    # Unnecessary but just for sanity.
    assert len(x) == sum(map(len, subsets))


DEFAULT_DISTRIBUTION_TRAIN = dict(
    ntimesteps=['const', 1],
    batchsz=['const', 8],
    # imsize=['const', 360],
    # weight_decay=['log_uniform_format', 1e-6, 1e-2, '.2g'],
    optimizer=['choice', ['momentum', 'adam']],
    momentum=['one_minus_log_uniform_format', 0.8, 0.99, '.2g'],
    use_nesterov=['choice', [True, False]],
    adam_beta1=['one_minus_log_uniform_format', 0.8, 0.99, '.2g'],
    adam_beta2=['one_minus_log_uniform_format', 0.99, 0.9999, '.2g'],
    adam_epsilon=['log_uniform_format', 1e-9, 1, '.2g'],
    lr_init=['log_uniform_format', 1e-6, 1e-2, '.2g'],
    lr_decay_rate=['choice', [1, 0.5, 0.1]],
    lr_decay_steps=['choice', [10000, 100000, 100000]],
    grad_clip=['choice', [True, False]],
    max_grad_norm=['log_uniform_format', 1e-3, 1e3, '.1g'],
    # num_steps=['const', 100000],
    sampler_kind=['choice', ['sampling', 'regular', 'freq-range-fit']],
    sampler_freq=['log_uniform_format', 5, 200, '.2g'],
    sampler_center_freq=['log_uniform_format', 5, 200, '.2g'],
    sampler_relative_freq_range=['uniform_format', 0, 1, '.2f'],
    sampler_use_log=['choice', [True, False]],
)

DEFAULT_DISTRIBUTION_SIAMFC = dict(
    # Options required for determining stride of network.
    feature_padding=['const', 'VALID'],
    feature_arch=['choice', ['alexnet', 'darknet']],
    increase_stride=['choice', [[], [2], [4], [2, 2], [1, 2]]],
    desired_template_size=['choice', [96, 128]],
    desired_relative_search_size=['choice', [1.5, 2, 3]],
    template_scale=['uniform_format', 1, 3, '.2g'],
    aspect_method=['choice', ['perimeter', 'area', 'stretch']],
    use_gt=['const', True],
    curr_as_prev=['const', True],
    pad_with_mean=['const', True],
    feather=['const', True],
    feather_margin=['log_uniform_format', 0.01, 0.1, '.1g'],
    center_input_range=['const', True],
    keep_uint8_range=['const', False],
    feature_act=['const', 'linear'],
    enable_feature_bnorm=['const', True],
    enable_template_mask=['choice', [True, False]],
    xcorr_padding=['const', 'VALID'],
    bnorm_after_xcorr=['const', True],
    freeze_siamese=['const', False],
    learnable_prior=['const', False],
    train_multiscale=['const', False],
    # Tracking parameters:
    num_scales=['choice', [3, 5, 7]],
    # scale_step=1.03,
    max_scale_delta=['log_uniform_format', 0.03, 0.3, '.2g'],
    scale_update_rate=['choice', [0, 0.5]],
    report_square=['const', False],
    hann_method=['const', 'add_logit'],  # none, mul_prob, add_logit
    hann_coeff=['log_uniform_format', 0.1, 10, '.1g'],
    arg_max_eps_rel=['choice', [0, 0.02, 0.05, 0.1]],
    # Loss parameters:
    wd=['choice', [1e-4, 1e-6]],
    enable_ce_loss=['const', True],
    ce_label=['choice', ['gaussian_distance', 'best_iou']],
    ce_label_structure=['const', 'independent'],
    sigma=['choice', [0.1, 0.2, 0.3]],
    balance_classes=['choice', [True, False]],
    enable_margin_loss=['const', False],
    margin_cost=['choice', ['iou', 'iou_correct', 'distance']],
    margin_reduce_method=['choice', ['max', 'mean']],
)

DEFAULT_DISTRIBUTION = dict()
DEFAULT_DISTRIBUTION.update(DEFAULT_DISTRIBUTION_TRAIN)
DEFAULT_DISTRIBUTION.update(DEFAULT_DISTRIBUTION_SIAMFC)

KEYS_TRAIN = DEFAULT_DISTRIBUTION_TRAIN.keys()
KEYS_SIAMFC = DEFAULT_DISTRIBUTION_SIAMFC.keys()


def sample_vector_train(rand, p):
    '''
    The keys of the vector will be the same.
    Some values may be set to None.
    '''
    def sample(spec):
        return search.sample_param(rand, *spec)

    x = {k: None for k in KEYS_TRAIN}
    x['ntimesteps'] = sample(p['ntimesteps'])
    x['batchsz'] = sample(p['batchsz'])
    # x['imsize'] = sample(p['imsize'])
    # x['weight_decay'] = sample(p['weight_decay'])
    x['optimizer'] = sample(p['optimizer'])
    if x['optimizer'] == 'momentum':
        x['momentum'] = sample(p['momentum'])
        x['use_nesterov'] = sample(p['use_nesterov'])
    elif x['optimizer'] == 'adam':
        x['adam_beta1'] = sample(p['adam_beta1'])
        x['adam_beta2'] = sample(p['adam_beta2'])
        x['adam_epsilon'] = sample(p['adam_epsilon'])
    x['lr_init'] = sample(p['lr_init'])
    x['lr_decay_rate'] = sample(p['lr_decay_rate'])
    if x['lr_decay_rate'] != 1:
        x['lr_decay_steps'] = sample(p['lr_decay_steps'])
    x['grad_clip'] = sample(p['grad_clip'])
    if x['grad_clip']:
        x['max_grad_norm'] = sample(p['max_grad_norm'])
    # x['num_steps'] = sample(p['num_steps'])

    x['sampler_kind'] = sample(p['sampler_kind'])
    if x['sampler_kind'] == 'sampling':
        pass
    elif x['sampler_kind'] == 'regular':
        x['sampler_freq'] = sample(p['sampler_freq'])
    elif x['sampler_kind'] == 'freq-range-fit':
        x['sampler_center_freq'] = sample(p['sampler_center_freq'])
        x['sampler_relative_freq_range'] = sample(p['sampler_relative_freq_range'])
        x['sampler_use_log'] = sample(p['sampler_use_log'])
    return x


def sample_vector_siamfc(rand, p):
    '''
    The keys of the vector will be the same.
    Some values may be set to None.
    '''
    def sample(spec):
        return search.sample_param(rand, *spec)

    x = {k: None for k in KEYS_SIAMFC}

    x['feature_padding'] = sample(p['feature_padding'])
    x['feature_arch'] = sample(p['feature_arch'])
    x['increase_stride'] = sample(p['increase_stride'])
    x['desired_template_size'] = sample(p['desired_template_size'])
    x['desired_relative_search_size'] = sample(p['desired_relative_search_size'])
    x['template_scale'] = sample(p['template_scale'])
    x['aspect_method'] = sample(p['aspect_method'])
    x['use_gt'] = sample(p['use_gt'])
    # TODO: Condition on use_gt?
    x['curr_as_prev'] = sample(p['curr_as_prev'])
    x['pad_with_mean'] = sample(p['pad_with_mean'])
    x['feather'] = sample(p['feather'])
    if x['feather']:
        x['feather_margin'] = sample(p['feather_margin'])
    x['center_input_range'] = sample(p['center_input_range'])
    x['keep_uint8_range'] = sample(p['keep_uint8_range'])
    x['feature_act'] = sample(p['feature_act'])
    x['enable_feature_bnorm'] = sample(p['enable_feature_bnorm'])
    x['enable_template_mask'] = sample(p['enable_template_mask'])
    x['xcorr_padding'] = sample(p['xcorr_padding'])
    x['bnorm_after_xcorr'] = sample(p['bnorm_after_xcorr'])
    x['freeze_siamese'] = sample(p['freeze_siamese'])
    x['learnable_prior'] = sample(p['learnable_prior'])
    x['train_multiscale'] = sample(p['train_multiscale'])

    # Tracking parameters:
    x['num_scales'] = sample(p['num_scales'])
    if x['num_scales'] > 1:
        x['max_scale_delta'] = sample(p['max_scale_delta'])
        x['scale_update_rate'] = sample(p['scale_update_rate'])
    x['report_square'] = sample(p['report_square'])
    x['hann_method'] = sample(p['hann_method'])
    if x['hann_method'] == 'add_logit':
        x['hann_coeff'] = sample(p['hann_coeff'])
    x['arg_max_eps_rel'] = sample(p['arg_max_eps_rel'])

    # Loss parameters:
    x['wd'] = sample(p['wd'])
    x['enable_ce_loss'] = sample(p['enable_ce_loss'])
    if x['enable_ce_loss']:
        x['ce_label'] = sample(p['ce_label'])
        x['ce_label_structure'] = sample(p['ce_label_structure'])
        x['sigma'] = sample(p['sigma'])
        x['balance_classes'] = sample(p['balance_classes'])
    x['enable_margin_loss'] = sample(p['enable_margin_loss'])
    if x['enable_margin_loss']:
        x['margin_cost'] = sample(p['margin_cost'])
        x['margin_reduce_method'] = sample(p['margin_reduce_method'])

    return x


def train_vector_to_kwargs(x):
    '''
    Args:
        x: Dict with keys KEYS_TRAIN.

    Returns:
        Dict of kwargs for train.train().
    '''
    kwargs = dict(x)
    info = {}

    # del kwargs['momentum']
    # del kwargs['use_nesterov']
    # del kwargs['adam_beta1']
    # del kwargs['adam_beta2']
    # del kwargs['adam_epsilon']
    # optimizer_params = {}
    # if x['optimizer'] == 'momentum':
    #     optimizer_params['momentum'] = x['momentum']
    #     optimizer_params['use_nesterov'] = x['use_nesterov']
    # elif x['optimizer'] == 'adam':
    #     optimizer_params['beta1'] = x['adam_beta1']
    #     optimizer_params['beta2'] = x['adam_beta2']
    #     optimizer_params['epsilon'] = x['adam_epsilon']
    # kwargs['optimizer_params'] = optimizer_params

    del kwargs['sampler_kind']
    del kwargs['sampler_freq']
    del kwargs['sampler_center_freq']
    del kwargs['sampler_relative_freq_range']
    del kwargs['sampler_use_log']
    kwargs['sampler_params'] = {'kind': x['sampler_kind']}
    if x['sampler_kind'] == 'sampling':
        pass
    elif x['sampler_kind'] == 'regular':
        kwargs['sampler_params']['freq'] = x['sampler_freq']
    elif x['sampler_kind'] == 'freq-range-fit':
        center_freq = x['sampler_center_freq']
        radius = x['sampler_relative_freq_range']
        min_freq = max(1, (1 - radius) * center_freq)
        kwargs['sampler_params']['min_freq'] = min_freq
        if not x['sampler_use_log']:
            max_freq = center_freq + (center_freq - min_freq)
        else:
            log_center_freq = math.log(center_freq)
            log_min_freq = math.log(min_freq)
            log_max_freq = log_center_freq + (log_center_freq - log_min_freq)
            max_freq = math.exp(log_max_freq)
        kwargs['sampler_params']['max_freq'] = max_freq
        kwargs['sampler_params']['use_log'] = x['sampler_use_log']

    return kwargs, info


def siamfc_vector_to_kwargs(x):
    '''
    Args:
        x: Dict with keys KEYS_SIAMFC.

    Returns:
        Dict of kwargs for SiamFC.__init__().
    '''
    kwargs = dict(x)
    info = {}

    # Replace max_scale_delta with scale_step.
    del kwargs['max_scale_delta']
    if x['num_scales'] > 1:
        assert x['num_scales'] % 2 == 1
        max_scale = (x['num_scales'] - 1) / 2
        # (1 + max_scale_delta) = scale_step ** max_scale
        # scale_step = (1 + max_scale_delta) ** (1 / max_scale)
        scale_step = (1 + x['max_scale_delta']) ** (1. / max_scale)
    else:
        scale_step = None
    kwargs['scale_step'] = scale_step
    info['scale_step'] = scale_step

    # Replace desired_xxx parameters with final.
    del kwargs['desired_template_size']
    del kwargs['desired_relative_search_size']
    offset, stride = _compute_rf(x['feature_arch'],
                                 x['feature_padding'],
                                 x['increase_stride'])
    template_size = _round_lattice(x['desired_template_size'], offset, stride)
    desired_search_size = x['desired_relative_search_size'] * template_size
    search_size = _round_lattice(desired_search_size, offset, stride)
    # Log these intermediate values.
    info.update(dict(desired_search_size=desired_search_size,
                     offset=offset, stride=stride,
                     template_size=template_size, search_size=search_size))
    kwargs['template_size'] = template_size
    kwargs['search_size'] = search_size

    return kwargs, info


def _compute_rf(arch, padding, increase_stride):
    # TODO: Avoid the copy-paste here?
    # TODO: Use increase_stride!
    rfs = {'input': cnnutil.identity_rf()}
    if arch == 'alexnet':
        _, rfs = _alexnet(None, rfs, padding)
    elif arch == 'darknet':
        _, rfs = _darknet(None, rfs, padding)
    rf = rfs['input']
    # Ensure uniform in x and y.
    assert all(x == rf.stride[0] for x in rf.stride)
    assert all(x == rf.rect.min[0] for x in rf.rect.min)
    assert all(x == rf.rect.max[0] for x in rf.rect.max)
    rf.stride = rf.stride[0]
    rf.rect.min = rf.rect.min[0]
    rf.rect.max = rf.rect.max[0]
    # Receptive field of first output pixel is [min, max).
    # Receptive field of n output pixels is [min, max + stride * (n - 1)).
    # For this to be centered in [0, len), we need
    #   min - 0 = len - [max + stride * (n - 1)]
    #   min + max + stride * (n - 1) = len
    # If this looks a bit funny, think of it as 2 * (max + min) / 2.
    # The value offset is also the minimum size.
    # If min is negative due to padding,
    # then max + min will be smaller than the receptive field.
    offset = rf.rect.min + rf.rect.max
    return offset, rf.stride


def _alexnet(x, rfs, padding):
    '''x can be None'''
    x, rfs = util.conv2d_rf(x, rfs, 96, [11, 11], 2, padding=padding, scope='conv1')
    x, rfs = util.max_pool2d_rf(x, rfs, [3, 3], 2, padding=padding, scope='pool1')
    x, rfs = util.conv2d_rf(x, rfs, 256, [5, 5], padding=padding, scope='conv2')
    x, rfs = util.max_pool2d_rf(x, rfs, [3, 3], 2, padding=padding, scope='pool2')
    x, rfs = util.conv2d_rf(x, rfs, 384, [3, 3], padding=padding, scope='conv3')
    x, rfs = util.conv2d_rf(x, rfs, 384, [3, 3], padding=padding, scope='conv4')
    # x, rfs = util.conv2d_rf(x, rfs, 256, [3, 3], padding=padding, scope='conv5',
    #                         activation_fn=get_act(output_act), normalizer_fn=None)
    x, rfs = util.conv2d_rf(x, rfs, 256, [3, 3], padding=padding, scope='conv5')
    return x, rfs


def _darknet(x, rfs, padding):
    '''x can be None'''
    # with slim.arg_scope([slim.conv2d], activation_fn=leaky_relu):
    x, rfs = util.conv2d_rf(None, rfs, 16, [3, 3], 1, padding=padding, scope='conv1')
    x, rfs = util.max_pool2d_rf(None, rfs, [3, 3], 2, padding=padding, scope='pool1')
    x, rfs = util.conv2d_rf(None, rfs, 32, [3, 3], 1, padding=padding, scope='conv2')
    x, rfs = util.max_pool2d_rf(None, rfs, [3, 3], 2, padding=padding, scope='pool2')
    x, rfs = util.conv2d_rf(None, rfs, 64, [3, 3], 1, padding=padding, scope='conv3')
    x, rfs = util.max_pool2d_rf(None, rfs, [3, 3], 2, padding=padding, scope='pool3')
    x, rfs = util.conv2d_rf(None, rfs, 128, [3, 3], 1, padding=padding, scope='conv4')
    x, rfs = util.max_pool2d_rf(None, rfs, [3, 3], 2, padding=padding, scope='pool4')
    # x, rfs = util.conv2d_rf(x, rfs, 256, [3, 3], 1, padding=padding, scope='conv5',
    #                         activation_fn=get_act(output_act), normalizer_fn=None)
    x, rfs = util.conv2d_rf(None, rfs, 256, [3, 3], 1, padding=padding, scope='conv5')
    return x, rfs


def _round_lattice(x, offset, stride):
    multiple = int(round(float(x - offset) / stride))
    return max(0, multiple) * stride + offset
