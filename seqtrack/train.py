import pdb
import sys
import csv
import itertools
import json
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug
import time
import os
import random
import re
import subprocess
import threading

from seqtrack import draw
from seqtrack import evaluate
from seqtrack import geom
from seqtrack import graph
from seqtrack import helpers
from seqtrack import motion
from seqtrack import pipeline
from seqtrack import sample
from seqtrack import visualize

from seqtrack.helpers import cache_json


def train(model, datasets, eval_sets, o, stat=None, use_queues=False):
    '''Trains a network.

    Args:
        create_model: Function that takes as input a dictionary of tensors and
            returns a model object.
        datasets: Dictionary of datasets with keys 'train' and 'val'.
        eval_sets: A dictionary of sampling functions which return collections
            of sequences on which to evaluate the tracker.

    Returns:
        The results obtained using the tracker at different stages of training.
        This is a dictionary with the same keys as eval_sets:
            all_results[val_set] = list of (iter_num, results)
        Note that this is represented as a list of pairs instead of a dictionary
        to facilitate saving to JSON (does not support integer keys).

    The reason that the model is provided as a *function* is so that
    the code which uses the model is free to decide how to instantiate it.
    For example, training code may construct a single instance of the model with input placeholders,
    or it may construct two instances of the model, each with its own input queue.

    The model should use `tf.get_variable` rather than `tf.Variable` to facilitate variable sharing between multiple instances.
    The model will be used in the same manner as an input to `tf.make_template`.

    The input dictionary has fields::

        'x0'         # First image in sequence, shape [b, h, w, 3]
        'y0'         # Position of target in first image, shape [b, 4]
        'x'          # Input images, shape [b, n, h, w, 3]
        'y_is_valid' # Booleans indicating presence of frame, shape [b, n]
        'aspect'     # Aspect ratio (width/height) of original image, shape [b]

    and the output dictionary has fields::

        'y'       # (optional) Predicted position of target in each frame, shape [b, n, 4]
        'heatmap' # (optional) Score for pixel belonging to target, shape [b, n, h, w, 1]

    The images provided to the model are already normalized (e.g. dataset mean subtracted).

    Each sampler in the eval_sets dictionary is a function that
    returns a collection of sequences or is a finite generator of sequences.
    '''

    # How should we compute training and validation error with pipelines?
    # Option 1 is to have multiple copies of the network with shared parameters.
    # However, this makes it difficult to plot training and validation error on the same axes
    # since there are separate summary ops for training and validation (with different names).
    # Option 2 is to use FIFOQueue.from_list()

    modes = ['train', 'val']

    feed_loop = {} # Each value is a function to call in a thread.
    with tf.name_scope('input'):
        from_queue = None
        if use_queues:
            queues = []
            for mode in modes:
                # Create a queue each for training and validation data.
                queue, feed_loop[mode] = _make_input_pipeline(
                    ntimesteps=o.ntimesteps, batchsz=o.batchsz, im_size=(o.imheight, o.imwidth),
                    num_load_threads=1, num_batch_threads=1, name='pipeline_'+mode)
                queues.append(queue)
            queue_index, from_queue = pipeline.make_multiplexer(queues,
                capacity=4, num_threads=1)
        example, run_opts = graph.make_placeholders(
            o.ntimesteps, (o.imheight, o.imwidth), default=from_queue)

    # color augmentation
    example = _perform_color_augmentation(example, o)

    # Always use same statistics for whitening (not set dependent).
    ## stat = datasets['train'].stat
    # TODO: Mask y with use_gt to prevent accidental use.
    # model = create_model(graph.whiten(graph.guard_labels(example), stat=stat),
    #                      summaries_collections=['summaries_model'])
    # loss_var, model.gt = get_loss(example, model.outputs, model.gt, o)
    # example_input = graph.whiten(graph.guard_labels(example), stat=stat)
    example_input = graph.whiten(example, stat=stat)
    with tf.name_scope('model'):
        outputs, loss_var, init_state, final_state = model.instantiate(
            example_input, run_opts, enable_loss=True,
            image_summaries_collections=['IMAGE_SUMMARIES'])
    r = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.scalar('regularization', r)
    loss_var += r
    tf.summary.scalar('total', loss_var)

    _draw_summaries(example, outputs)

    nepoch     = o.nepoch if not o.debugmode else 2
    nbatch     = len(datasets['train'].videos)/o.batchsz if not o.debugmode else 30
    nbatch_val = len(datasets['val'].videos)/o.batchsz if not o.debugmode else 30

    global_step_var = tf.Variable(0, name='global_step', trainable=False)
    # lr = init * decay^(step)
    #    = init * decay^(step / period * period / decay_steps)
    #    = init * [decay^(period / decay_steps)]^(step / period)
    lr = tf.train.exponential_decay(o.lr_init, global_step_var,
                                    decay_steps=o.lr_decay_steps,
                                    decay_rate=o.lr_decay_rate,
                                    staircase=True)
    tf.summary.scalar('lr', lr, collections=['summaries_train'])
    optimizer = _get_optimizer(lr, o)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if not o.grad_clip:
            optimize_op = optimizer.minimize(loss_var, global_step=global_step_var)
        else: # Gradient clipping by norm; NOTE: `global graident clipping` may be another correct way.
            gradients, variables = zip(*optimizer.compute_gradients(loss_var))
            gradients = [None if gradient is None else tf.clip_by_norm(gradient, o.max_grad_norm)
                         for gradient in gradients]
            optimize_op = optimizer.apply_gradients(zip(gradients, variables),
                                                    global_step=global_step_var)

    summary_vars = {}
    summary_vars_with_preview = {}
    global_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    with tf.name_scope('summary'):
        image_summaries = tf.get_collection('IMAGE_SUMMARIES')
        for mode in modes:
            with tf.name_scope(mode):
                # Merge summaries with any that are specific to the mode.
                summaries = (global_summaries + tf.get_collection('summaries_' + mode))
                summary_vars[mode] = tf.summary.merge(summaries)
                summaries.extend(image_summaries)
                # Assume that model summaries could contain images.
                # TODO: Separate model summaries into image and non-image.
                summaries.extend(tf.get_collection('summaries_model'))
                summary_vars_with_preview[mode] = tf.summary.merge(summaries)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=10)

    # Use a separate random number generator for each sampler.
    sequences = {mode: iter_examples(datasets[mode], o,
                                     rand=np.random.RandomState(o.seed_global),
                                     num_epochs=None)
                 for mode in modes}

    if o.curriculum_learning:
        ''' Curriculum learning.
        Restore values of trainable variables from pre-trained model on short sequence,
        to initialize and train a model on longer sequences.
        Note that since I define restoring variables from `trainable variables`
        in the current model, if the pre-trained model doesn't have those variables,
        it will fail to restore by the saver.
        '''
        vars_to_restore = list(tf.trainable_variables())
        saver_cl = tf.train.Saver(vars_to_restore)

    if o.cnn_pretrain:
        ''' In case of loading pre-trained CNN (e.g., vgg_16), create a separate
        Saver object that is going to be used to restore when session starts.
        '''
        #from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        #print_tensors_in_checkpoint_file('./pretrained/vgg_16.ckpt', None, False)
        # or
        #from tensorflow.python import pywrap_tensorflow
        #reader = pywrap_tensorflow.NewCheckpointReader('./pretrained/vgg_16.ckpt')
        #var_to_shape_map = reader.get_variable_to_shape_map()
        # Approach 1. Use tf.trainable_variables won't work if variables are non-trainable.
        #vars_to_restore = {v.name.split(':')[0]: v for v in tf.trainable_variables()
        #                   if o.cnn_model in v.name}
        # Approach 2. Use collection to get variables.
        vars_to_restore = {v.name.split(':')[0]: v for v in tf.get_collection(o.cnn_model)}
        saver_external = tf.train.Saver(vars_to_restore)

    t_total = time.time()
    with tf.Session(config=o.tfconfig) as sess:
        print '\ntraining starts! --------------------------------------------'
        sys.stdout.flush()

        # 1. resume (full restore), 2. initialize from scratch, 3. curriculume learning (partial restore)
        prev_ckpt = 0
        if o.resume:
            model_file = tf.train.latest_checkpoint(o.path_ckpt)
            if model_file is None:
                raise ValueError('could not find checkpoint')
            print 'restore: {}'.format(model_file)
            saver.restore(sess, model_file)
            print 'done: restore'
            sys.stdout.flush()
            prev_ckpt = global_step_var.eval()
        elif o.cnn_pretrain:
            model_file = os.path.join(o.path_data_home, 'pretrained', '{}.ckpt'.format(o.cnn_model))
            saver_external.restore(sess, model_file)
            #print sess.run(tf.report_uninitialized_variables()) # To check
            # initialize uninitialized variables
            vars_uninit = sess.run(tf.report_uninitialized_variables())
            sess.run(tf.variables_initializer([v for v in tf.global_variables()
                                               if v.name.split(':')[0] in vars_uninit]))
            assert len(sess.run(tf.report_uninitialized_variables())) == 0
        else:
            sess.run(init_op)
            if o.curriculum_learning:
                if o.pretrained_cl is None: # e.g., '/some_path/ckpt/iteration-150000'
                    raise ValueError('could not find checkpoint')
                print 'restore: {}'.format(o.pretrained_cl)
                saver_cl.restore(sess, o.pretrained_cl)
                print 'done: (partial) restore for curriculum learning'
                sys.stdout.flush()

        if use_queues:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess, coord)
            # Run the feed loops in another thread.
            threads = [threading.Thread(target=feed_loop[mode],
                                        args=(sess, coord, sequences[mode]))
                       for mode in modes]
            for t in threads:
                t.start()

        if o.tfdb:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        writer = {}
        for mode in modes:
            path_summary = os.path.join(o.path_summary, mode)
            # Only include graph in one summary.
            if mode == 'train':
                writer[mode] = tf.summary.FileWriter(path_summary, sess.graph)
            else:
                writer[mode] = tf.summary.FileWriter(path_summary)

        while True: # Loop over epochs
            global_step = global_step_var.eval() # Number of steps taken.
            if global_step >= nepoch * nbatch:
                break
            ie = global_step / nbatch
            t_epoch = time.time()

            ib_val = 0
            loss_ep = []
            for ib in range(nbatch): # Loop over batches in epoch.
                global_step = global_step_var.eval() # Number of steps taken.

                if not o.nosave:
                    period_ckpt = o.period_ckpt if not o.debugmode else 40
                    if global_step % period_ckpt == 0 and global_step > prev_ckpt:
                        if not os.path.isdir(o.path_ckpt):
                            os.makedirs(o.path_ckpt)
                        print 'save model'
                        saver.save(sess, o.path_ckpt+'/iteration', global_step=global_step)
                        print 'done: save model'
                        sys.stdout.flush()
                        prev_ckpt = global_step

                # intermediate evaluation of model
                period_assess = o.period_assess if not o.debugmode else 20
                if global_step > 0 and global_step > o.period_skip and global_step % period_assess == 0:
                    iter_id = 'iteration{}'.format(global_step)
                    for eval_id, sampler in eval_sets.iteritems():
                        vis_dir = os.path.join(o.path_output, iter_id, eval_id)
                        if not os.path.isdir(vis_dir): os.makedirs(vis_dir, 0755)
                        # visualizer = visualize.VideoFileWriter(vis_dir)
                        # Run the tracker on a full epoch.
                        print 'evaluation: {}'.format(eval_id)
                        eval_sequences = sampler()
                        # eval_sequences = [
                        #     motion.augment(sequence, rand=np.random,
                        #         translate_kind='laplace',
                        #         translate_amount=0.1,
                        #         scale_kind='laplace',
                        #         scale_exp_amount=math.exp(math.log(1.01)/30.))
                        #     for sequence in eval_sequences
                        # ]
                        # Cache the results.
                        result_file = os.path.join(o.path_output, 'assess', eval_id,
                            iter_id+'.json')
                        result = cache_json(result_file,
                            lambda: evaluate.evaluate(sess, example, model, eval_sequences,
                                # visualize=visualizer.visualize if o.save_videos else None,
                                visualize=True, vis_dir=vis_dir, save_frames=o.save_frames,
                                use_gt=o.use_gt_eval, tre_num=o.eval_tre_num),
                            makedir=True)
                        assert 'OPE' in result
                        modes = [mode for mode in ['OPE', 'TRE'] if mode in result]
                        for mode in modes:
                            print 'mode {}: IOU: {:.3f}, AUC: {:.3f}, CLE: {:.3f}, Prec.@20px: {:.3f}'.format(
                                mode, result[mode]['iou_mean'], result[mode]['auc'],
                                result[mode]['cle_mean'], result[mode]['cle_representative'])

                # Take a training step.
                start = time.time()
                feed_dict = {run_opts['use_gt']:      o.use_gt_train,
                             run_opts['is_training']: True,
                             run_opts['is_tracking']: False,
                             run_opts['gt_ratio']:    max(1.0*np.exp(-o.gt_decay_rate*global_step),
                                                         o.min_gt_ratio)}
                if use_queues:
                    feed_dict.update({queue_index: 0}) # Choose validation queue.
                else:
                    batch_seqs = [next(sequences['train']) for i in range(o.batchsz)]
                    # batch = _load_batch(batch_seqs, o)
                    batch = graph.load_batch(batch_seqs, o.ntimesteps, (o.imheight, o.imwidth))
                    feed_dict.update({example[k]: v for k, v in batch.iteritems()})
                    dur_load = time.time() - start
                if global_step % o.period_summary == 0:
                    summary_var = (summary_vars_with_preview['train']
                                   if global_step % o.period_preview == 0
                                   else summary_vars['train'])
                    _, loss, summary = sess.run([optimize_op, loss_var, summary_var],
                                                feed_dict=feed_dict)
                    dur = time.time() - start
                    writer['train'].add_summary(summary, global_step=global_step)
                else:
                    _, loss = sess.run([optimize_op, loss_var], feed_dict=feed_dict)
                    dur = time.time() - start
                loss_ep.append(loss)

                newval = False
                # Evaluate validation error.
                if global_step % o.period_summary == 0:
                    # Only if (ib / nbatch) >= (ib_val / nbatch_val), or equivalently
                    if ib * nbatch_val >= ib_val * nbatch:
                        start = time.time()
                        feed_dict = {run_opts['use_gt']:      o.use_gt_train,  # Match training.
                                     run_opts['is_training']: False, # Do not update bnorm stats.
                                     run_opts['is_tracking']: False,
                                     run_opts['gt_ratio']:    max(1.0*np.exp(-o.gt_decay_rate*ie), o.min_gt_ratio)} # Match training.
                        if use_queues:
                            feed_dict.update({queue_index: 1}) # Choose validation queue.
                        else:
                            batch_seqs = [next(sequences['val']) for i in range(o.batchsz)]
                            # batch = _load_batch(batch_seqs, o)
                            batch = graph.load_batch(batch_seqs, o.ntimesteps, (o.imheight, o.imwidth))
                            feed_dict.update({example[k]: v for k, v in batch.iteritems()})
                            dur_load = time.time() - start
                        summary_var = (summary_vars_with_preview['val']
                                       if global_step % o.period_preview == 0
                                       else summary_vars['val'])
                        loss_val, summary = sess.run([loss_var, summary_var],
                                                     feed_dict=feed_dict)
                        dur_val = time.time() - start
                        writer['val'].add_summary(summary, global_step=global_step)
                        ib_val += 1
                        newval = True

                # Print result of one batch update
                if o.verbose_train:
                    losstime = '|loss:{:.5f}/{:.5f} (time:{:.2f}/{:.2f}) - with val'.format(
                            loss, loss_val, dur, dur_val) if newval else \
                            '|loss:{:.5f} (time:{:.2f})'.format(loss, dur)
                    print 'ep {}/{}, batch {}/{} (bsz:{}), global_step {} {}'.format(
                            ie+1, nepoch, ib+1, nbatch, o.batchsz, global_step, losstime)
                    sys.stdout.flush()

            print '[Epoch finished] ep {:d}/{:d}, global_step {:d} |loss:{:.5f} (time:{:.2f})'.format(
                    ie+1, nepoch, global_step_var.eval(), np.mean(loss_ep), time.time()-t_epoch)
            sys.stdout.flush()

        # **training finished
        print '\ntraining finished! ------------------------------------------'
        print 'total time elapsed: {0:.2f}'.format(time.time()-t_total)


def _get_optimizer(lr, o):
    if o.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif o.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)
    elif o.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
    elif o.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    else:
        raise ValueError('optimizer not implemented or simply wrong.')
    return optimizer


def _make_input_pipeline(ntimesteps, batchsz, im_size, dtype=tf.float32,
        example_capacity=4, load_capacity=4, batch_capacity=4,
        num_load_threads=1, num_batch_threads=1,
        name='pipeline'):
    '''
    Args:
        im_size: (height, width) to construct tensor
    '''
    with tf.name_scope(name) as scope:
        height, width = im_size
        files, feed_loop = pipeline.get_example_filenames(capacity=example_capacity)
        images = pipeline.load_images(files, capacity=load_capacity,
                num_threads=num_load_threads, image_size=[height, width, 3])
        images_batch = pipeline.batch(images,
            batch_size=batchsz, sequence_length=ntimesteps+1,
            capacity=batch_capacity, num_threads=num_batch_threads)

        # Set static dimension of sequence length.
        # TODO: This may only be necessary due to how the model is written.
        images_batch['images'].set_shape([None, ntimesteps+1, None, None, None])
        images_batch['labels'].set_shape([None, ntimesteps+1, None])
        # Cast type of images.
        images_batch['images'] = tf.cast(images_batch['images'], tf.float32)
        # Put in format expected by model.
        # is_valid = (range(1, ntimesteps+1) < tf.expand_dims(images_batch['num_frames'], -1))
        example_batch = {
            'x0':         images_batch['images'][:, 0],
            'y0':         images_batch['labels'][:, 0],
            'x':          images_batch['images'][:, 1:],
            'y':          images_batch['labels'][:, 1:],
            'y_is_valid': images_batch['label_is_valid'][:, 1:],
            'aspect':     images_batch['aspect'],
        }
        return example_batch, feed_loop


def _perform_color_augmentation(example_raw, o, name='color_augmentation'):

    example = dict(example_raw)

    xs_aug = tf.concat([tf.expand_dims(example['x0'], 1), example['x']], 1)

    if o.color_augmentation.get('brightness', False):
        xs_aug = tf.image.random_brightness(xs_aug, 0.1)

    if o.color_augmentation.get('contrast', False):
        xs_aug = tf.image.random_contrast(xs_aug, 0.1, 1.1)

    if o.color_augmentation.get('grayscale', False):
        max_grayscale_ratio = 0.2
        rand_prob = tf.random_uniform(shape=[], minval=0, maxval=1)
        xs_aug = tf.cond(tf.less_equal(rand_prob, max_grayscale_ratio),
                         lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(xs_aug)),
                         lambda: xs_aug)

    example['x0'] = xs_aug[:,0]
    example['x']  = xs_aug[:,1:]
    return example


def iter_examples(dataset, o, rand=None, num_epochs=None):
    '''Generator that produces multiple epochs of examples for SGD.'''
    if num_epochs:
        epochs = xrange(num_epochs)
    else:
        epochs = itertools.count()
    for i in epochs:
        sequences = sample.sample(dataset, rand=rand,
                                  shuffle=True, max_objects=1, ntimesteps=o.ntimesteps,
                                  **o.sampler_params)
        for sequence in sequences:
            # JV: Add motion augmentation.
            # yield sequence
            if o.augment_motion:
                sequence = motion.augment(sequence, rand=rand, **o.motion_params)
            yield sequence


def generate_report(samplers, datasets, o,
        modes=['OPE', 'TRE'],
        metrics=['iou_mean', 'auc']):
    '''Finds all results for each evaluation distribution.

    Identifies the best result for each metric.
    Caution: More frequent evaluations might lead to better results.
    '''
    def helper():
        eval_id_fn = lambda sampler, dataset: '{}-{}'.format(dataset, sampler)
        best_fn = {'iou_mean': np.amax, 'auc': np.amax, 'cle_mean': np.amin, 'cle_representative': np.amax}
        report_dir = os.path.join(o.path_output, 'report')
        if not os.path.isdir(report_dir): os.makedirs(report_dir)

        # Plot each metric versus iteration.
        # Create one plot per sampler, with a line per dataset.
        for sampler in samplers:
            # Load all results using this sampler.
            results = {dataset: load_results(eval_id_fn(sampler, dataset)) for dataset in datasets}
            # Print results for each dataset.
            for dataset in datasets:
                for mode in modes:
                    print '==== evaluation: sampler {}, dataset {}, mode {} ===='.format(sampler, dataset, mode)
                    steps = sorted(results[dataset].keys())
                    for step in steps:
                        print 'iter {}:  {}'.format(step,
                            '; '.join(['{}: {:.3g}'.format(metric, results[dataset][step][mode][metric])
                                       for metric in metrics]))
                    for metric in metrics:
                        values = [results[dataset][step][mode][metric] for step in steps]
                        print 'best {}: {:.3g}'.format(metric, np.asscalar(best_fn[metric](values)))
            for mode in modes:
                # Generate plot for each metric.
                # Take union of steps for all datasets.
                steps = sorted(set.union(*[set(r.keys()) for r in results.values()]))
                for metric in metrics:
                    # Plot this metric over time for all datasets.
                    data_file = 'sampler-{}-mode-{}-metric-{}'.format(sampler, mode, metric)
                    with open(os.path.join(report_dir, data_file+'.tsv'), 'w') as f:
                        write_data_file(f, mode, metric, steps, results)
                    try:
                        plot_file = plot_data(report_dir, data_file)
                        print 'plot created:', plot_file
                    except Exception as e:
                        print 'could not create plot:', e

    def load_results(eval_id):
        '''Returns a dictionary from step number to dictionary of metrics.'''
        dirname = os.path.join(o.path_output, 'assess', eval_id)
        pattern = re.compile(r'^iteration(\d+)\.json$')
        results = {}
        for f in os.listdir(dirname):
            if not os.path.isfile(os.path.join(dirname, f)):
                continue
            match = pattern.match(f)
            if not match:
                continue
            step = int(match.group(1))
            with open(os.path.join(dirname, f), 'r') as r:
                results[step] = json.load(r)
        if not results:
            print 'warning: no results found:', eval_id
        return results

    def write_data_file(f, mode, metric, steps, results):
        # Create a column for the variance.
        fieldnames = ['step'] + [x+suffix for x in datasets for suffix in ['', '_std_err']]
        w = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        w.writeheader()
        for step in steps:
            # Map variance of metric to variance of 
            row = {
                dataset+suffix:
                    gnuplot_str(results[dataset].get(step, {}).get(mode, {}).get(metric+suffix, None))
                for dataset in datasets
                for suffix in ['', '_std_err']}
            row['step'] = step
            w.writerow(row)

    def plot_data(plot_dir, filename):
        src_dir = os.path.dirname(__file__)
        args = ['gnuplot',
            '-e', 'filename = "{}"'.format(filename),
            os.path.join(src_dir, 'plot_eval_metric.gnuplot'),
        ]
        p = subprocess.Popen(args, cwd=plot_dir)
        p.wait()
        return os.path.join(plot_dir, filename+'.png')

    return helper()

def gnuplot_str(x):
    if x is None:
        return '?'
    return str(x)


def _draw_summaries(example, outputs):
    with tf.name_scope('image_summary'):
        ntimesteps = example['x'].shape.as_list()[1]
        assert ntimesteps is not None

        # TODO: Use standard tf format so this is unnecessary.
        x0 = example['x0'][0] / 255.
        x = example['x'][0] / 255.

        tf.summary.image(
            'image_0', max_outputs=1,
            tensor=_draw_rectangles([x0], gt=[example['y0'][0]], dtype=tf.uint8))
        tf.summary.image(
            'image_1_to_T', max_outputs=ntimesteps,
            tensor=_draw_rectangles(
                x, gt=example['y'][0], gt_is_valid=example['y_is_valid'][0],
                pred=outputs['y'][0], dtype=tf.uint8))

def _draw_rectangles(im, gt, gt_is_valid=None, pred=None, dtype=None):
    if dtype is None:
        # Convert back to original dtype.
        dtype = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    if gt_is_valid is not None:
        gt = tf.where(gt_is_valid, gt, tf.zeros_like(gt) + geom.unit_rect())
    boxes = [gt]
    if pred is not None:
        boxes.append(pred)
    boxes = map(geom.rect_to_tf_box, boxes)
    im = tf.image.draw_bounding_boxes(im, tf.stack(boxes, axis=1))
    return tf.image.convert_image_dtype(im, dtype)
