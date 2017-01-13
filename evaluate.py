import pdb
import tensorflow as tf

import draw


def evaluate(sess, m, loader, o, data_=None):
    # TODO: add more evaluation criteria 

    if data_ == 'train':
        datasz = loader.ntr
    elif data_ == 'val':
        datasz = loader.nva
    elif data_ == 'test':
        datasz = loader.nte
    else:
        raise ValueError('no valid data kind')

    results = {'idx': [], 'inputs': [], 'labels': [], 'outputs': []}
    for ib in range(datasz/o.batchsz if not o.debugmode else 100):
        batch, idx = loader.load_batch(ib, o, data_=data_)

        fdict = {
                m.net['inputs']: batch['inputs'],
                m.net['inputs_length']: batch['inputs_length'],
                m.net['labels']: batch['labels']
                }
        outputs = sess.run(m.net['outputs'], feed_dict=fdict)

        results['idx'].append(idx)
        results['inputs'].append(batch['inputs'])
        results['labels'].append(batch['labels'])
        results['outputs'].append(outputs)

    # TODO: add more evalutation criteria (eg., IOU)
    # TODO: maybe separate eval wrapper, considering change of dataset
    # TODO: saving option
    draw.show_tracking_results_moving_mnist(results, o) 


