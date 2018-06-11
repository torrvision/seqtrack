from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
import sys
import tempfile

import logging
logger = logging.getLogger(__name__)

from seqtrack import train


def train_worker(name, **kwargs):
    pprint.pprint(kwargs, file=sys.stderr)
    tmp_dir = _get_tmp_dir()
    # Set args that can only be set at run-time or that depend on the name.
    output = train.train(dir=os.path.join('trials', name),
                         summary_dir='summary', summary_name=name,
                         tmp_data_dir=os.path.join(tmp_dir, 'data'),
                         **kwargs)
    pprint.pprint(output, file=sys.stderr)
    return output


def _get_tmp_dir():
    if _is_slurm_job():
        # TODO: This is specific to our cluster. Make general!
        return '/raid/local_scratch/{}-{}'.format(
            os.environ['SLURM_JOB_USER'], os.environ['SLURM_JOB_ID'])
    else:
        return tempfile.mkdtemp()


def _is_slurm_job():
    return 'SLURM_JOB_ID' in os.environ
