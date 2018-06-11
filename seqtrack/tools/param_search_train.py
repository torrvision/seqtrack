from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import json
import msgpack
import numpy as np
import os
import pprint

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import cnnutil
from seqtrack import helpers
from seqtrack import search
from seqtrack import train
from seqtrack.models import util

# The pickled object must be imported to unpickle in a different package (slurmproc.worker).
from seqtrack.tools import param_search_train_work as work


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    input_stream = make_input_stream(args, np.random.RandomState(args.seed))

    search.main(
        work.train_worker,
        input_stream,
        kwargs_fn=functools.partial(to_kwargs, args),
        postproc_fn=functools.partial(select_best_epoch, args),
        use_existing_inputs=True,
        max_num_configs=args.num_configs,
        report_only=args.report_only,
        cache_dir='cache',
        input_codec='json', kwargs_codec='json', output_codec='msgpack',
        use_slurm=args.use_slurm,
        slurm_flags=['--' + x for x in args.slurm_flags])


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', default='info', help='debug, info, warning')

    parser.add_argument('--seed', type=int, default=0, help='seed for configurations')
    parser.add_argument('--distribution', type=json.loads,
                        help='Override distribution of a parameter (JSON string)')
    parser.add_argument('--distribution_file',
                        help='Override distribution of a parameter (JSON file)')

    parser.add_argument('-n', '--num_configs', type=int, default=10,
                        help='number of configurations')
    parser.add_argument('--report_only', action='store_true')
    parser.add_argument('--no_slurm', dest='use_slurm', action='store_false',
                        help='Submit jobs to slurm or run directly?')
    parser.add_argument('--slurm_flags', nargs='+', help='flags for sbatch (without "--")')

    app.add_setup_data_args(parser)
    app.add_tracker_config_args(parser)
    app.add_eval_args(parser)
    # Keep image resolution fixed across trials.
    parser.add_argument('--imwidth', type=int, default=360, help='image resolution')
    parser.add_argument('--imheight', type=int, default=360, help='image resolution')
    parser.add_argument('--num_steps', type=int, default=100000,
                        help='Total number of gradient steps')
    parser.add_argument('--verbose_train', action='store_true')

    parser.add_argument('--optimize_dataset', default='pool_val-full',
                        help='eval_dataset to use to choose model')
    parser.add_argument('--optimize_metric', default='TRE_3_iou_seq_mean',
                        help='metric to optimize for')

    return parser.parse_args()


DEFAULT_DISTRIBUTION_TRAIN = dict(
    dataset=['choice', ['ilsvrc', 'ytbb', 'ilsvrc_ytbb']],
    ilsvrc_frac=['choice', [0.3, 0.5, 0.7]],
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
    # sampler_kind=['choice', ['sampling', 'regular', 'freq-range-fit']],
    sampler_kind=['choice', ['sampling']],  # Until FPS is implemented!
    sampler_freq=['log_uniform_format', 5, 200, '.2g'],
    sampler_center_freq=['log_uniform_format', 5, 200, '.2g'],
    sampler_relative_freq_range=['uniform_format', 0, 1, '.2f'],
    sampler_use_log=['choice', [True, False]],
)

DEFAULT_DISTRIBUTION_SIAMFC = dict(
    # Options required for determining stride of network.
    feature_padding=['const', 'VALID'],
    feature_arch=['choice', ['alexnet', 'darknet']],
    increase_stride=['const', []],
    desired_template_size=['choice', [96, 128, 192]],
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
    template_mask_kind=['choice', ['none', 'static', 'dynamic']],
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
    ce_pos_weight=['choice', [0.1, 1, 10]],
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


def make_input_stream(args, rand):
    p = dict(DEFAULT_DISTRIBUTION)
    if args.distribution_file:
        with open(args.distribution_file, 'r') as f:
            file_contents = json.load(f)
        helpers.assert_key_subset(file_contents, p)
        p.update(file_contents)
    if args.distribution:
        helpers.assert_key_subset(args.distribution, p)
        p.update(args.distribution)

    while True:
        vector = sample_vector(rand, p)
        name = search.hash_vector(vector)
        yield name, vector


def sample_vector(rand, p):
    helpers.assert_partition(set(p.keys()), [KEYS_TRAIN, KEYS_SIAMFC])
    p_train = {k: p[k] for k in KEYS_TRAIN}
    p_model = {k: p[k] for k in KEYS_SIAMFC}
    x_train = sample_vector_train(rand, p_train)
    x_model = sample_vector_siamfc(rand, p_model)
    x = {}
    x.update(x_train)
    x.update(x_model)
    return x


def sample_vector_train(rand, p):
    '''
    The keys of the vector will be the same.
    Some values may be set to None.
    '''
    def sample(spec):
        return search.sample_param(rand, *spec)

    x = {k: None for k in KEYS_TRAIN}

    x['dataset'] = sample(p['dataset'])
    if x['dataset'] == 'ilsvrc_ytbb':
        x['ilsvrc_frac'] = sample(p['ilsvrc_frac'])
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
    x['template_mask_kind'] = sample(p['template_mask_kind'])
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
        x['ce_pos_weight'] = sample(p['ce_pos_weight'])
        x['sigma'] = sample(p['sigma'])
        x['balance_classes'] = sample(p['balance_classes'])
    x['enable_margin_loss'] = sample(p['enable_margin_loss'])
    if x['enable_margin_loss']:
        x['margin_cost'] = sample(p['margin_cost'])
        x['margin_reduce_method'] = sample(p['margin_reduce_method'])

    return x


def to_kwargs(args, vector):
    helpers.assert_partition(set(vector.keys()), [KEYS_TRAIN, KEYS_SIAMFC])
    train_vector = {k: vector[k] for k in KEYS_TRAIN}
    model_vector = {k: vector[k] for k in KEYS_SIAMFC}
    train_kwargs, train_info = to_kwargs_train(args, train_vector)
    model_kwargs, model_info = to_kwargs_siamfc(args, model_vector)

    train_kwargs['model_params'] = model_kwargs
    # Merge info dicts.
    train_info.update(model_info)
    # return train_kwargs, train_info
    return train_kwargs


def to_kwargs_train(args, x):
    '''
    Args:
        x: Dict with keys KEYS_TRAIN.

    Returns:
        Dict of kwargs for train.train().
    '''
    # kwargs = dict(x)
    kwargs = {}
    info = {}

    # dir,
    # model_params,
    kwargs['seed'] = 0

    # Dataset:
    # train_dataset=None,
    # val_dataset=None,
    if x['dataset'] in ['ilsvrc', 'ytbb']:
        kwargs['train_dataset'] = x['dataset'] + '_train'
        kwargs['val_dataset'] = x['dataset'] + '_val'
    elif x['dataset'] == 'ilsvrc_ytbb':
        p_ilsvrc = x['ilsvrc_frac']
        kwargs['train_dataset'] = [[p_ilsvrc, 'ilsvrc_train'], [1 - p_ilsvrc, 'ytbb_train']]
        kwargs['val_dataset'] = [[p_ilsvrc, 'ilsvrc_val'], [1 - p_ilsvrc, 'ytbb_val']]
    kwargs['eval_datasets'] = args.eval_datasets
    kwargs['pool_datasets'] = args.pool_datasets
    kwargs['pool_split'] = args.pool_split
    kwargs['untar'] = args.untar
    kwargs['data_dir'] = args.data_dir
    kwargs['tar_dir'] = args.tar_dir
    # tmp_data_dir=None,
    kwargs['preproc_id'] = args.preproc
    kwargs['data_cache_dir'] = args.data_cache_dir

    # Sampling:
    # sampler_params=None,
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
    # augment_motion=None,
    # motion_params=None,

    # Evaluation:
    kwargs['eval_samplers'] = args.eval_samplers
    kwargs['max_eval_videos'] = args.max_eval_videos

    # Training process:
    kwargs['ntimesteps'] = x['ntimesteps']

    # resume=False,
    # tfdb=False,
    kwargs['use_queues'] = args.use_queues
    # summary_dir=None,
    # summary_name=None,
    kwargs['nosave'] = args.nosave
    kwargs['period_ckpt'] = args.period_ckpt
    kwargs['period_assess'] = args.period_assess
    kwargs['period_skip'] = args.period_skip
    kwargs['period_summary'] = args.period_summary
    kwargs['period_preview'] = args.period_preview
    kwargs['verbose_train'] = args.verbose_train
    kwargs['visualize'] = args.visualize
    kwargs['keep_frames'] = args.keep_frames
    kwargs['session_config_kwargs'] = dict(gpu_manctrl=args.gpu_manctrl, gpu_frac=args.gpu_frac,
                                           log_device_placement=args.log_device_placement)

    # Training args:
    # ntimesteps=None,
    kwargs['batchsz'] = x['batchsz']
    kwargs['imwidth'] = args.imwidth
    kwargs['imheight'] = args.imheight
    kwargs['lr_init'] = x['lr_init']
    kwargs['lr_decay_steps'] = x['lr_decay_steps']
    kwargs['lr_decay_rate'] = x['lr_decay_rate']
    kwargs['optimizer'] = x['optimizer']
    # optimizer_params = {}
    # if x['optimizer'] == 'momentum':
    #     optimizer_params['momentum'] = x['momentum']
    #     optimizer_params['use_nesterov'] = x['use_nesterov']
    # elif x['optimizer'] == 'adam':
    #     optimizer_params['beta1'] = x['adam_beta1']
    #     optimizer_params['beta2'] = x['adam_beta2']
    #     optimizer_params['epsilon'] = x['adam_epsilon']
    # kwargs['optimizer_params'] = optimizer_params
    kwargs['momentum'] = x['momentum']
    kwargs['use_nesterov'] = x['use_nesterov']
    kwargs['adam_beta1'] = x['adam_beta1']
    kwargs['adam_beta2'] = x['adam_beta2']
    kwargs['adam_epsilon'] = x['adam_epsilon']
    # kwargs['weight_decay'] = x['weight_decay']
    kwargs['grad_clip'] = x['grad_clip']
    kwargs['max_grad_norm'] = x['max_grad_norm']
    # siamese_pretrain=None,
    # siamese_model_file=None,
    kwargs['num_steps'] = args.num_steps
    # use_gt_train=None,
    # gt_decay_rate=None,
    # min_gt_ratio=None,

    # Evaluation args:
    # use_gt_eval=False,
    kwargs['eval_tre_num'] = args.eval_tre_num

    return kwargs, info


def to_kwargs_siamfc(args, x):
    '''
    Args:
        x: Dict with keys KEYS_SIAMFC.

    Returns:
        Dict of kwargs for SiamFC.__init__().
    '''
    # kwargs = dict(x)
    kwargs = {}
    info = {}

    # template_size=127,
    # search_size=255,
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
    kwargs['template_scale'] = x['template_scale']
    kwargs['aspect_method'] = x['aspect_method']
    kwargs['use_gt'] = x['use_gt']
    kwargs['curr_as_prev'] = x['curr_as_prev']
    kwargs['pad_with_mean'] = x['pad_with_mean']
    kwargs['feather'] = x['feather']
    kwargs['feather_margin'] = x['feather_margin']
    kwargs['center_input_range'] = x['center_input_range']
    kwargs['keep_uint8_range'] = x['keep_uint8_range']
    kwargs['feature_padding'] = x['feature_padding']
    kwargs['feature_arch'] = x['feature_arch']
    kwargs['increase_stride'] = x['increase_stride']
    kwargs['feature_act'] = x['feature_act']
    kwargs['enable_feature_bnorm'] = x['enable_feature_bnorm']
    kwargs['template_mask_kind'] = x['template_mask_kind']
    kwargs['xcorr_padding'] = x['xcorr_padding']
    kwargs['bnorm_after_xcorr'] = x['bnorm_after_xcorr']
    kwargs['freeze_siamese'] = x['freeze_siamese']
    kwargs['learnable_prior'] = x['learnable_prior']
    kwargs['train_multiscale'] = x['train_multiscale']

    # Tracking parameters:
    kwargs['search_method'] = 'local'
    # global_search_min_resolution=64,
    # global_search_max_resolution=512,
    # global_search_num_scales=4, # 64, 128, 256, 512
    kwargs['num_scales'] = x['num_scales']
    # scale_step=1.03,
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
    kwargs['scale_update_rate'] = x['scale_update_rate']
    kwargs['report_square'] = x['report_square']
    kwargs['hann_method'] = x['hann_method']
    kwargs['hann_coeff'] = x['hann_coeff']
    kwargs['arg_max_eps_rel'] = x['arg_max_eps_rel']

    # Loss parameters:
    kwargs['wd'] = x['wd']
    kwargs['enable_ce_loss'] = x['enable_ce_loss']
    kwargs['ce_label'] = x['ce_label']
    kwargs['ce_pos_weight'] = x['ce_pos_weight']
    kwargs['ce_label_structure'] = x['ce_label_structure']
    kwargs['sigma'] = x['sigma']
    kwargs['balance_classes'] = x['balance_classes']
    kwargs['enable_margin_loss'] = x['enable_margin_loss']
    kwargs['margin_cost'] = x['margin_cost']
    kwargs['margin_reduce_method'] = x['margin_reduce_method']

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


def select_best_epoch(args, results):
    # Use optimize_dataset to choose checkpoint of each experiment.
    # If there were multiple trials, take mean over trials.
    return train.summarize_trials([results], val_dataset=args.optimize_dataset,
                                  sort_key=lambda metrics: metrics[args.optimize_metric])


if __name__ == '__main__':
    main()
