import pdb
import sys
import numpy as np
import tensorflow as tf
import threading
import time
import os

import draw
from evaluate import evaluate
import helpers


def create_track_examples(loader, set_name):
    str_feature        = lambda x: tf.train.Feature(bytes_list=tf.train.BytesList(value=x))
    int_list_feature   = lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=x))
    float_list_feature = lambda x: tf.train.Feature(float_list=tf.train.FloatList(value=x))
    int_feature        = lambda x: int_list_feature([x])
    float_feature      = lambda x: float_list_feature([x])

    tracks = []
    for i in range(loader.nsnps[set_name]):
        image_dir = loader.snps['train']['Data'][i]
        print image_dir
        annotations_dir = loader.snps['train']['Annotations'][i]
        video_len = loader.nfrms_snp[set_name][i]
        objs = loader.objids_snp[set_name][i]
        for obj in range(len(objs)):
            print obj
            # # Find frames that contain the object.
            # obj_frames = [t for t in range(video_len)
            #               if obj in loader.objids_allfrm_snp[set_name][i][t]]

            get_image_file = lambda t: os.path.join(image_dir, '{:06d}.JPEG'.format(t))
            image_file_list = map(get_image_file, range(video_len))
            # List of length 1 or 0 depending on object presence.
            xmin = [[] for t in range(video_len)]
            xmax = [[] for t in range(video_len)]
            ymin = [[] for t in range(video_len)]
            ymax = [[] for t in range(video_len)]

            track = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    'length': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[video_len])),
                    # 'label_frames': tf.train.Feature(
                    #     int64_list=tf.train.Int64List(value=obj_frames)),
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    'image_file': tf.train.FeatureList(feature=map(str_feature, image_file_list)),
                    'xmin':       tf.train.FeatureList(feature=map(float_list_feature, xmin)),
                    'xmax':       tf.train.FeatureList(feature=map(float_list_feature, xmax)),
                    'ymin':       tf.train.FeatureList(feature=map(float_list_feature, ymin)),
                    'ymax':       tf.train.FeatureList(feature=map(float_list_feature, ymax)),
                }))
            yield track


def train(m, loader, o):
    nepoch     = o.nepoch if not o.debugmode else 2
    nbatch     = loader.nexps['train']/o.batchsz if not o.debugmode else 30
    nbatch_val = loader.nexps['val']/o.batchsz if not o.debugmode else 30

    global_step_var = tf.Variable(0, name='global_step', trainable=False)
    # lr = init * decay^(step)
    #    = init * decay^(step / period * period / decay_steps)
    #    = init * [decay^(period / decay_steps)]^(step / period)
    lr = tf.train.exponential_decay(o.lr_init, global_step_var,
                                    decay_steps=o.lr_decay_steps,
                                    decay_rate=o.lr_decay_rate,
                                    staircase=True)
    optimizer = _get_optimizer(lr, o)
    optimize_op = optimizer.minimize(m.net['loss'], global_step=global_step_var)

    # # Enqueue a list of images (a tensor of strings).
    # snippet_images = []
    # for i in range(loader.nsnps['train']):
    #     image_dir = loader.snps['train']['Data'][i]
    #     video_len = loader.nfrms_snp['train'][i]
    #     get_image_file = lambda t: os.path.join(image_dir, '{:06d}.JPEG'.format(t))
    #     snippet_images.append(map(get_image_file, range(video_len)))

    QUEUE_CAPACITY = 32
    QUEUE_THREADS = 4

    example_files = tf.placeholder(tf.string, shape=[None,], name='example_files')
    example_labels = tf.placeholder(tf.float32, shape=[None,4], name='example_labels')
    # Queue contains (valid, image files, label, x0 file, y0)
    file_queue = tf.FIFOQueue(capacity=QUEUE_CAPACITY,
        dtypes=[tf.string, tf.float32],
        names=['files', 'labels'],
        name='file_queue')
    example_for_file_queue = {'files': example_files, 'labels': example_labels}
    enqueue_file_op = file_queue.enqueue(example_for_file_queue)
    print '-' * 40
    print 'example_for_file_queue:', example_for_file_queue

    example_from_file_queue = file_queue.dequeue()
    image_contents = tf.map_fn(tf.read_file, example_from_file_queue['files'], dtype=tf.string)
    images = tf.map_fn(tf.image.decode_jpeg, image_contents, dtype=tf.uint8)
    print '-' * 40
    print 'images:', images

    image_queue = tf.FIFOQueue(capacity=QUEUE_CAPACITY,
        dtypes=[tf.uint8, tf.float32],
        names=['images', 'labels'],
        name='image_queue')
    enqueue_image_op = image_queue.enqueue({'images': images, 'labels': example_from_file_queue['labels']})
    tf.train.add_queue_runner(
        tf.train.QueueRunner(image_queue, [enqueue_image_op] * QUEUE_THREADS))

    example_from_image_queue = image_queue.dequeue()
    print '-' * 40
    print 'example_from_image_queue:', example_from_image_queue

    # TODO: Check if a mistake here raises an error.
    example_from_image_queue['images'].set_shape([None, None, None, 3])
    example_from_image_queue['labels'].set_shape([None, 4])
    print '-' * 40
    print 'example_from_image_queue:', example_from_image_queue

    full_example = {
        'images_raw': example_from_image_queue['images'],
        'label':      example_from_image_queue['labels'],
        'x0_raw':     example_from_image_queue['images'][0],
        'y0':         example_from_image_queue['labels'][0],
    }
    print '-' * 40
    print 'full_example:', full_example
    print '-' * 40
    # TODO: Get length of sequences before padding?
    batch = tf.train.batch(full_example, batch_size=o.batchsz, dynamic_pad=True)
    print '-' * 40
    print 'batch:', batch

    init_op = tf.global_variables_initializer()
    with tf.Session(config=o.tfconfig) as sess:
        def enqueue_batch():
            for i in range(loader.nexps['train']):
                example = loader.get_example(i, o, dstype='train')
                sess.run(enqueue_file_op, feed_dict={example_files: example['files'],
                                                     example_labels: example['labels']})

        sess.run(init_op)

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord)
        t = threading.Thread(target=enqueue_batch)
        t.start()

        # sess.run(enqueue_file_op)
        # images_val = sess.run(images)
        batch_val = sess.run(batch)
        pdb.set_trace()

    # ----------------------------------------

    records_file = 'tracks_train.tfrecords'
    if not os.path.exists(records_file):
        print 'generate track examples'
        tracks = list(create_track_examples(loader, 'train'))
        # tracks = [example.SerializeToString() for example in tracks]

        print 'write tracks to file'
        with tf.python_io.TFRecordWriter(records_file) as writer:
            for track in tracks:
                writer.write(track.SerializeToString())

    filename_queue = tf.train.string_input_producer([records_file])
    result = tf.TFRecordReader().read(filename_queue)
    serialized_example = result[1]

    # queue = tf.train.string_input_producer(tracks)
    context_features, sequence_features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={
            'length': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        },
        sequence_features={
            'image_file': tf.VarLenFeature(dtype=tf.string),
            # 'xmin':       tf.VarLenSequenceFeature(dtype=tf.int64),
        })
    print
    print 'context_features:'
    print context_features
    print
    print 'sequence_features:'
    print sequence_features

    image_files_var = sequence_features['image_file']
    print
    print 'image_files_var:'
    print image_files_var

    # # Read each image once.
    # image_file_queue = tf.train.string_input_producer(image_files_var, num_epochs=1)
    # _, image_content = tf.WholeFileReader().read(image_file_queue)
    # image_var = tf.image.decode_jpeg(image_content)

    # # def f(dataset):
    # #     num_label_frames = tf.constant(dataset.num_label_frames)
    # #     video_index = tf.train.range_input_producer(dataset.num_videos)
    # #     label_frames = tf.constant(pad(dataset.label_frames[video_index]))
    # #     first_frame = tf.multinomial(tf.log())

    # # # (manual) learning rate recipe
    # # lr_recipe = _get_lr_recipe()

    init_op = tf.global_variables_initializer()
    summary_var_eval = tf.summary.merge_all()
    # Optimization summary might include gradients, learning rate, etc.
    summary_var_opt = tf.summary.merge([summary_var_eval,
        tf.summary.scalar('lr', lr)])
    saver = tf.train.Saver()

    with tf.Session(config=o.tfconfig) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        image_files = sess.run([image_files_var])
        print
        print 'image_files:'
        print image_files
        pdb.set_trace()
        # image = sess.run([image_var])

    # ----------------------------------------

    def process_batch(batch, step, optimize=True, writer=None, write_summary=False):
        names = ['target_raw', 'inputs_raw', 'x0_raw', 'y0', 'inputs_valid', 'inputs_HW', 'labels']
        fdict = {m.net[name]: batch[name] for name in names}
        # if optimize:
        #     fdict.update({
        #         lr: lr_epoch,
        #     })
        start = time.time()
        summary = None
        if optimize:
            if write_summary:
                _, loss, summary = sess.run([optimize_op, m.net['loss'], summary_var_opt], feed_dict=fdict)
            else:
                _, loss = sess.run([optimize_op, m.net['loss']], feed_dict=fdict)
        else:
            if write_summary:
                loss, summary = sess.run([m.net['loss'], summary_var_eval], feed_dict=fdict)
            else:
                loss = sess.run([m.net['loss']], feed_dict=fdict)
        dur = time.time() - start
        if write_summary:
            writer.add_summary(summary, global_step=step)
        return loss, dur


    t_total = time.time()
    with tf.Session(config=o.tfconfig) as sess:
        # Either initialize or restore model.
        if o.resume:
            model_file = tf.train.latest_checkpoint(o.path_ckpt)
            print "restore: {}".format(model_file)
            saver.restore(sess, model_file)
        else:
            sess.run(init_op)


        path_summary_train = os.path.join(o.path_summary, 'train')
        path_summary_val = os.path.join(o.path_summary, 'val')
        train_writer = tf.summary.FileWriter(path_summary_train, sess.graph)
        val_writer = tf.summary.FileWriter(path_summary_val)

        while True: # Loop over epochs
            global_step = global_step_var.eval() # Number of steps taken.
            if global_step >= nepoch * nbatch:
                break
            ie = global_step / nbatch
            t_epoch = time.time()
            loader.update_epoch_begin('train')
            loader.update_epoch_begin('val')
            # lr_epoch = lr_recipe[ie] if o.lr_update else o.lr

            ib_val = 0
            for ib in range(nbatch): # Loop over batches in epoch.
                global_step = global_step_var.eval() # Number of steps taken.

                if not o.nosave:
                    period_ckpt = o.period_ckpt if not o.debugmode else 40
                    if global_step > 0 and global_step % period_ckpt == 0: # save intermediate model
                        if not os.path.isdir(o.path_ckpt):
                            os.makedirs(o.path_ckpt)
                        fname = os.path.join(o.path_ckpt, 'iteration{}.ckpt'.format(global_step))
                        # saved_model = saver.save(sess, fname)
                        saver.save(sess, fname)

                # **after a certain iteration, perform the followings
                # - evaluate on train/test/val set
                # - print results (loss, eval resutls, time, etc.)
                period_assess = o.period_assess if not o.debugmode else 20
                if global_step > 0 and global_step % period_assess == 0: # evaluate model
                    print ' '
                    # evaluate
                    val_ = 'test' if o.dataset == 'bouncing_mnist' else 'val'
                    evals = {
                        'train': evaluate(sess, m, loader, o, 'train',
                            np.maximum(int(np.floor(100/o.batchsz)), 1),
                            hold_inputs=True, shuffle_local=True),
                        val_: evaluate(sess, m, loader, o, val_,
                            np.maximum(int(np.floor(100/o.batchsz)), 1),
                            hold_inputs=True, shuffle_local=True)}
                    # visualize tracking results examples
                    draw.show_track_results(
                        evals['train'], loader, 'train', o, global_step,nlimit=20)
                    draw.show_track_results(
                        evals[val_], loader, val_, o, global_step,nlimit=20)
                    # print results
                    print 'ep {:d}/{:d} (STEP-{:d}) '\
                        '|(train/{:s}) IOU: {:.3f}/{:.3f}, '\
                        'AUC: {:.3f}/{:.3f}, CLE: {:.3f}/{:.3f} '.format(
                        ie+1, nepoch, global_step+1, val_,
                        evals['train']['iou_mean'], evals[val_]['iou_mean'],
                        evals['train']['auc'],      evals[val_]['auc'],
                        evals['train']['cle_mean'], evals[val_]['cle_mean'])

                # Take a training step.
                start = time.time()
                batch = loader.get_batch(ib, o, dstype='train')
                load_dur = time.time() - start

                loss, dur = process_batch(batch, step=global_step, optimize=True,
                    writer=train_writer,
                    write_summary=(ib % o.summary_period == 0))

                # **results after every batch
                print ('ep {0:d}/{1:d}, batch {2:d}/{3:d} (BATCH:{4:d}) '
                    '|loss:{5:.5f} |time:{6:.2f} ({7:.2f})').format(
                    ie+1, nepoch, ib+1, nbatch, o.batchsz, loss, dur, load_dur)

                # Evaluate validation error.
                if ib % o.val_period == 0:
                    # Only if (ib / nbatch) >= (ib_val / nbatch_val), or equivalently
                    if ib * nbatch_val >= ib_val * nbatch:
                        start = time.time()
                        batch = loader.get_batch(ib_val, o, dstype='val')
                        load_dur = time.time() - start
                        loss, dur = process_batch(batch, step=global_step, optimize=False,
                            writer=val_writer, write_summary=True)
                        print ('[val] ep {0:d}/{1:d}, batch {2:d}/{3:d} (BATCH:{4:d}) '
                            '|loss:{5:.5f} |time:{6:.2f} ({7:.2f})').format(
                            ie+1, nepoch, ib+1, nbatch, o.batchsz, loss, dur, load_dur)
                        ib_val += 1

            print 'ep {0:d}/{1:d} (EPOCH) |time:{2:.2f}'.format(
                    ie+1, nepoch, time.time()-t_epoch)

        # **training finished
        print '\ntraining finished! ------------------------------------------'
        print 'total time elapsed: {0:.2f}'.format(time.time()-t_total)


# def _get_lr_recipe():
#     # TODO: may need a different recipe; also consider exponential decay
#     # (previous) lr_epoch = o.lr*(0.1**np.floor(float(ie)/(nepoch/2))) \
#             #if o.lr_update else o.lr
#     # manual learning rate recipe
#     lr_recipe = np.zeros([100], dtype=np.float32)
#     for i in range(lr_recipe.shape[0]):
#         if i < 5:
#             lr_recipe[i] = 0.0001*(0.1**i) # TODO: check if this is alright
#         else:
#             lr_recipe[i] = lr_recipe[4]
#     return lr_recipe

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
