'''
Pipelines enable asynchronous image loading for faster training and evaluation.

Example::

    files, feed_loop = pipeline.get_example_filenames()
    images = pipeline.load_images(files)
    batch = pipeline.batch(images)
    # Get an iterable list of examples:
    examples = train.iter_examples(loader, o, dstype='train')
    # Define a function of the batch of images:
    op = f(batch)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord)
        t = threading.Thread(target=feed_loop, args=(sess, coord, examples))
        t.start()
        # Run an iterative procedure:
        for i in range(num_iters):
            sess.run(op)

The functions `get_example_filenames` and `load_images` each construct a queue.
Both functions internally define an operation to push data on to the queue,
and return an operation that pops data off the queue.
The structure of these functions is modeled on `tf.train.batch`.

There is one key difference between the two.
The function `load_images` gets its input from a tensor of filenames.
Since this is a pure TensorFlow operation, concurrency can be achieved by adding a queue runner to the graph.
The function `get_example_filenames` instead gets its input from a placeholder.
This must be manually fed using a loop running in a Python thread.
'''

import functools
import tensorflow as tf


def get_example_filenames(capacity=32, name='get_example'):
    '''Creates a queue of sequences.
    Each sequence contains a list of the filenames of its images.

    Each element of the queue represents a single example sequence.
    For a sequence of length n, each element is a dictionary with the elements::
        'image_files' # Tensor with shape [n] containing strings.
        'labels'      # Tensor with shape [n, 4] containing rectangles.

    Args:
        capacity: The size of the queue.

    Returns:
        A tuple (`example`, `feed_loop`), where
        `example` is a dictionary of tensors taken from the queue, and
        `feed_loop` is a partial evaluation of the function `feed_example_filenames`
        with the `placeholder` and `enqueue` parameters set.
        See the package example.
    '''
    with tf.name_scope(name) as scope:
        # Create queue to write examples to.
        queue = tf.FIFOQueue(capacity=capacity,
                             dtypes=[tf.string, tf.float32],
                             names=['image_files', 'labels'],
                             name='file_queue')
        placeholder = {
            'image_files': tf.placeholder(tf.string, shape=[None], name='example_files'),
            'labels':      tf.placeholder(tf.float32, shape=[None, 4], name='example_labels'),
        }
        enqueue = queue.enqueue(placeholder)
        with tf.name_scope('summary'):
            tf.summary.scalar('fraction_of_%d_full' % capacity,
                              tf.cast(queue.size(), tf.float32) * (1./capacity))
        dequeue = queue.dequeue(name=scope)

    # Restore partial shape information that is erased by FIFOQueue.
    for k in dequeue:
        dequeue[k].set_shape(placeholder[k].shape)

    feed_loop = functools.partial(feed_example_filenames, placeholder, enqueue)
    return dequeue, feed_loop


def feed_example_filenames(placeholder, enqueue, sess, coord, examples):
    '''Enqueues examples in a loop.
    This should be run in another thread.

    Args:
        placeholder: Dictionary of placeholder tensors.
        sess: tf.Session
        coord: tf.train.Coordinator
        examples: Iterable collection of dictionaries to feed.

    The dictionaries in `examples` will be fed into the `placeholder` tensors.
    Both dictionaries should have elements::

        'image_files' # List of image filenames.
        'labels'      # Numpy array with shape [n, 4] containing rectangles.

    The function `get_example_filenames` returns a function that calls this function.
    '''
    for example in examples:
        if coord.should_stop():
            return
        sess.run(enqueue, feed_dict={
            placeholder['image_files']:  example['image_files'],
            placeholder['labels']: example['labels'],
        })
    coord.request_stop()


def load_images(example, capacity=32, num_threads=1, image_size=[None, None, None],
        name='load_images'):
    '''Creates a queue of sequences with images loaded.
    See the package example.

    Args:
        example: Dictionary of input tensors.

    Returns:
        Dictionary of output tensors.

    The input dictionary has fields::

        'image_files' # Tensor with shape [n] containing strings.
        'labels'      # Tensor with shape [n, 4] containing rectangles.

    The output dictionary has fields::

        'images' # Tensor with shape [n, h, w, 3] containing images.
        'labels' # Tensor with shape [n, 4] containing rectangles.

    This function adds a queue runner to the graph.
    It is necessary to call `start_queue_runners` for this queue to work.
    '''
    # Follow structure of tf.train.batch().
    with tf.name_scope(name) as scope:
        # Create queue to write images to.
        queue = tf.FIFOQueue(capacity=capacity,
                             dtypes=[tf.uint8, tf.float32],
                             names=['images', 'labels'],
                             name='image_queue')
        example = dict(example)
        # Read files from disk.
        file_contents = tf.map_fn(tf.read_file, example['image_files'], dtype=tf.string)
        # Decode images.
        images = tf.map_fn(tf.image.decode_jpeg, file_contents, dtype=tf.uint8)
        # Replace files with images.
        del example['image_files']
        example['images'] = images
        enqueue = queue.enqueue(example)
        tf.train.add_queue_runner(tf.train.QueueRunner(queue, [enqueue]*num_threads))
        with tf.name_scope('summary'):
            tf.summary.scalar('fraction_of_%d_full' % capacity,
                              tf.cast(queue.size(), tf.float32) * (1./capacity))
        dequeue = queue.dequeue(name=scope)
        # Restore rank information of Tensors for tf.train.batch.
        # TODO: It does not seem possible to preserve this through the FIFOQueue?
        # Let at least the sequence length remain dynamic.
        dequeue['images'].set_shape([None] + image_size)
        dequeue['labels'].set_shape(example['labels'].shape)
        return dequeue

def batch(example, batch_size=1, capacity=32, num_threads=1, name='batch'):
    '''Creates a queue of batches of sequences (with images) from a queue of sequences.
    Pads all sequences to the maximum length and adds a field for the length of the sequence.
    See the package example.

    The input dictionary has fields::

        'images' # Tensor with shape [n, h, w, 3] containing images.
        'labels' # Tensor with shape [n, 4] containing rectangles.

    The output dictionary has fields::

        'images'     # Tensor with shape [b, n_max, h, w, 3] containing images.
        'labels'     # Tensor with shape [b, n_max, 4] containing rectangles.
        'num_frames' # Tensor with shape [b] containing the sequence length.
    '''
    with tf.name_scope(name) as scope:
        # Get the length of the sequence before tf.train.batch.
        example['num_frames'] = tf.shape(example['images'])[0]
        # TODO: This may produce batches of length < ntimesteps+1
        # since PaddingFIFOQueue pads to the *maximum* length.
        # Is this a problem for the training code?
        example_batch = tf.train.batch(example, batch_size=batch_size,
            dynamic_pad=True, capacity=capacity, num_threads=num_threads,
            name=scope)
    # Restore partial shape information that is erased by FIFOQueue.
    for k in example_batch:
        example_batch[k].set_shape([None] + example[k].shape.as_list())
    return example_batch


def make_multiplexer(sources, capacity=32, num_threads=1, name='multiplex'):
    # Take details from first source.
    names = sorted(sources[0].keys())
    dtypes = [sources[0][k].dtype for k in names]
    shapes = [sources[0][k].shape for k in names]
    with tf.name_scope(name) as scope:
        # Create dummy queues.
        queues = []
        for i, source in enumerate(sources):
            with tf.name_scope('dummy_{}'.format(i)):
                queue = tf.FIFOQueue(capacity=capacity, dtypes=dtypes, names=names)
                enqueue = queue.enqueue(source)
                with tf.name_scope('summary'):
                    tf.summary.scalar('fraction_of_%d_full' % capacity,
                                      tf.cast(queue.size(), tf.float32) * (1./capacity))
            tf.train.add_queue_runner(tf.train.QueueRunner(queue, [enqueue]*num_threads))
            queues.append(queue)
        # Multiplex over dummy queues.
        index = tf.placeholder(tf.int32, [], 'index')
        queue = tf.QueueBase.from_list(index, queues)
        example = queue.dequeue(name=scope)
    # Restore partial shape information that is erased by FIFOQueue.
    for k, shape in zip(names, shapes):
        example[k].set_shape(shape)
    return index, example
