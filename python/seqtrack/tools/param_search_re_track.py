'''Parameter search for tracking parameters.

Uses all checkpoints of a previously trained model.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import json
import numpy as np
import os

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import helpers
from seqtrack import param_search
from seqtrack.tools.param_search_train import _compute_rf
from seqtrack.tools.param_search_train import _round_lattice
from seqtrack.tools.param_search_train import select_best_epoch

# The pickled object must be imported to unpickle in a different package (slurmproc.worker).
from seqtrack.tools import param_search_train_work as work


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    input_stream = make_input_stream(args, np.random.RandomState(args.seed))

    with open(args.kwargs_file, 'r') as f:
        original_kwargs = json.load(f)  # TODO: Not assume JSON?

    param_search.main(
        functools.partial(work.train_worker,
                          only_evaluate_existing=True, override_ckpt_dir=args.ckpt_dir),
        input_stream,
        kwargs_fn=functools.partial(to_kwargs, original_kwargs),
        postproc_fn=functools.partial(select_best_epoch, args),
        use_existing_inputs=True,
        max_num_configs=args.num_configs,
        report_only=args.report_only,
        cache_dir='cache',
        input_codec='json', kwargs_codec='json', output_codec='msgpack',
        use_slurm=args.use_slurm,
        slurm_flags=['--' + x for x in args.slurm_flags],
        slurm_group_size=args.slurm_group_size)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', default='info', help='debug, info, warning')

    parser.add_argument('--kwargs_file', help='Original kwargs for training model')
    parser.add_argument('--ckpt_dir', help='Training directory that contains ckpt dir')

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
    parser.add_argument('--slurm_group_size', type=int, default=1)

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


DEFAULT_DISTRIBUTION = dict(
    # Options required for determining stride of network.
    # feature_padding=['const', 'VALID'],
    # feature_arch=['choice', ['alexnet', 'darknet']],
    # increase_stride=['const', []],
    # desired_template_size=['choice', [96, 128, 192]],
    desired_relative_search_size=['choice', [1.5, 2, 3]],
    # template_scale=['uniform_format', 1, 3, '.2g'],
    # aspect_method=['choice', ['perimeter', 'area', 'stretch']],
    # use_gt=['const', True],
    # curr_as_prev=['const', True],
    # pad_with_mean=['const', True],
    # feather=['const', True],
    # feather_margin=['log_uniform_format', 0.01, 0.1, '.1g'],
    # center_input_range=['const', True],
    # keep_uint8_range=['const', False],
    # feature_act=['const', 'linear'],
    # enable_feature_bnorm=['const', True],
    # template_mask_kind=['choice', ['none', 'static', 'dynamic']],
    # xcorr_padding=['const', 'VALID'],
    # bnorm_after_xcorr=['const', True],
    # freeze_siamese=['const', False],
    # learnable_prior=['const', False],
    # train_multiscale=['const', False],

    # Tracking parameters:
    num_scales=['choice', [3, 5, 7]],
    # scale_step=1.03,
    max_scale_delta=['log_uniform_format', 0.03, 0.3, '.2g'],
    scale_update_rate=['uniform_format', 0, 1, '.1f'],  # scale_update_rate=['choice', [0, 0.5]],
    report_square=['const', False],
    hann_method=['const', 'add_logit'],  # none, mul_prob, add_logit
    hann_coeff=['log_uniform_format', 0.1, 10, '.1g'],
    arg_max_eps_rel=['choice', [0, 0.02, 0.05, 0.1]],

    # Loss parameters:
    # wd=['choice', [1e-4, 1e-6]],
    # enable_ce_loss=['const', True],
    # ce_label=['choice', ['gaussian_distance', 'best_iou']],
    # ce_label_structure=['const', 'independent'],
    # ce_pos_weight=['choice', [0.1, 1, 10]],
    # sigma=['choice', [0.1, 0.2, 0.3]],
    # balance_classes=['choice', [True, False]],
    # enable_margin_loss=['const', False],
    # margin_cost=['choice', ['iou', 'iou_correct', 'distance']],
    # margin_reduce_method=['choice', ['max', 'mean']],
)

KEYS = DEFAULT_DISTRIBUTION.keys()


# TODO: Avoid copy paste?
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
        name = param_search.hash_vector(vector)
        yield name, vector


def sample_vector(rand, p):
    '''
    The keys of the vector will be the same.
    Some values may be set to None.
    '''
    def sample(spec):
        return param_search.sample_param(rand, *spec)

    x = {k: None for k in KEYS}

    # x['feature_padding'] = sample(p['feature_padding'])
    # x['feature_arch'] = sample(p['feature_arch'])
    # x['increase_stride'] = sample(p['increase_stride'])
    # x['desired_template_size'] = sample(p['desired_template_size'])
    x['desired_relative_search_size'] = sample(p['desired_relative_search_size'])
    # x['template_scale'] = sample(p['template_scale'])
    # x['aspect_method'] = sample(p['aspect_method'])
    # x['use_gt'] = sample(p['use_gt'])
    # TODO: Condition on use_gt?
    # x['curr_as_prev'] = sample(p['curr_as_prev'])
    # x['pad_with_mean'] = sample(p['pad_with_mean'])
    # x['feather'] = sample(p['feather'])
    # if x['feather']:
    #     x['feather_margin'] = sample(p['feather_margin'])
    # x['center_input_range'] = sample(p['center_input_range'])
    # x['keep_uint8_range'] = sample(p['keep_uint8_range'])
    # x['feature_act'] = sample(p['feature_act'])
    # x['enable_feature_bnorm'] = sample(p['enable_feature_bnorm'])
    # x['template_mask_kind'] = sample(p['template_mask_kind'])
    # x['xcorr_padding'] = sample(p['xcorr_padding'])
    # x['bnorm_after_xcorr'] = sample(p['bnorm_after_xcorr'])
    # x['freeze_siamese'] = sample(p['freeze_siamese'])
    # x['learnable_prior'] = sample(p['learnable_prior'])
    # x['train_multiscale'] = sample(p['train_multiscale'])

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
    # x['wd'] = sample(p['wd'])
    # x['enable_ce_loss'] = sample(p['enable_ce_loss'])
    # if x['enable_ce_loss']:
    #     x['ce_label'] = sample(p['ce_label'])
    #     x['ce_label_structure'] = sample(p['ce_label_structure'])
    #     x['ce_pos_weight'] = sample(p['ce_pos_weight'])
    #     x['sigma'] = sample(p['sigma'])
    #     x['balance_classes'] = sample(p['balance_classes'])
    # x['enable_margin_loss'] = sample(p['enable_margin_loss'])
    # if x['enable_margin_loss']:
    #     x['margin_cost'] = sample(p['margin_cost'])
    #     x['margin_reduce_method'] = sample(p['margin_reduce_method'])

    return x


def to_kwargs(original_kwargs, vector):
    kwargs = dict(original_kwargs)
    kwargs['model_params'] = to_kwargs_siamfc(original_kwargs['model_params'], vector)
    return kwargs


def to_kwargs_siamfc(original_kwargs, x):
    '''
    Args:
        x: Dict with keys KEYS_SIAMFC.

    Returns:
        Dict of kwargs for SiamFC.__init__().
    '''
    kwargs = dict(original_kwargs)
    info = {}

    # template_size=127,
    # search_size=255,
    offset, stride = _compute_rf(kwargs['feature_arch'], kwargs['feature_padding'],
                                 kwargs['increase_stride'])
    # template_size = _round_lattice(x['desired_template_size'], offset, stride)
    desired_search_size = x['desired_relative_search_size'] * kwargs['template_size']
    search_size = _round_lattice(desired_search_size, offset, stride)
    # Log these intermediate values.
    # info.update(dict(desired_search_size=desired_search_size,
    #                  offset=offset, stride=stride,
    #                  template_size=template_size, search_size=search_size))
    # kwargs['template_size'] = template_size
    kwargs['search_size'] = search_size
    # kwargs['template_scale'] = x['template_scale']
    # kwargs['aspect_method'] = x['aspect_method']
    # kwargs['use_gt'] = x['use_gt']
    # kwargs['curr_as_prev'] = x['curr_as_prev']
    # kwargs['pad_with_mean'] = x['pad_with_mean']
    # kwargs['feather'] = x['feather']
    # kwargs['feather_margin'] = x['feather_margin']
    # kwargs['center_input_range'] = x['center_input_range']
    # kwargs['keep_uint8_range'] = x['keep_uint8_range']
    # kwargs['feature_padding'] = x['feature_padding']
    # kwargs['feature_arch'] = x['feature_arch']
    # kwargs['increase_stride'] = x['increase_stride']
    # kwargs['feature_act'] = x['feature_act']
    # kwargs['enable_feature_bnorm'] = x['enable_feature_bnorm']
    # kwargs['template_mask_kind'] = x['template_mask_kind']
    # kwargs['xcorr_padding'] = x['xcorr_padding']
    # kwargs['bnorm_after_xcorr'] = x['bnorm_after_xcorr']
    # kwargs['freeze_siamese'] = x['freeze_siamese']
    # kwargs['learnable_prior'] = x['learnable_prior']
    # kwargs['train_multiscale'] = x['train_multiscale']

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
    # kwargs['wd'] = x['wd']
    # kwargs['enable_ce_loss'] = x['enable_ce_loss']
    # kwargs['ce_label'] = x['ce_label']
    # kwargs['ce_pos_weight'] = x['ce_pos_weight']
    # kwargs['ce_label_structure'] = x['ce_label_structure']
    # kwargs['sigma'] = x['sigma']
    # kwargs['balance_classes'] = x['balance_classes']
    # kwargs['enable_margin_loss'] = x['enable_margin_loss']
    # kwargs['margin_cost'] = x['margin_cost']
    # kwargs['margin_reduce_method'] = x['margin_reduce_method']

    # return kwargs, info
    return kwargs


if __name__ == '__main__':
    main()
