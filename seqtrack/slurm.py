from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import functools
import os
import tempfile

import logging
logger = logging.getLogger(__name__)

import slurmproc


class SlurmDictMapper(object):

    def __init__(self, poll_period=1, dir=None, tempdir=None, **kwargs):
        '''
        Args:
            dir: If none, then create a new directory under tempdir.
            tempdir: Only used if dir is none.
            kwargs: For slurmproc.Process.
        '''
        self._poll_period = poll_period
        self._kwargs = kwargs

        self._procs = {}  # Incomplete jobs. Dict that maps key to Process.
        self._errors = {}  # Dict that maps key to Exception.
        self._num = 0

        if not dir:
            assert tempdir
            if not os.path.exists(tempdir):
                os.makedirs(tempdir, 0o755)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
            dir = tempfile.mkdtemp(dir=tempdir, prefix=('tmp_{}_'.format(timestamp)))
        else:
            if os.path.exists(dir):
                raise RuntimeError('dir already exists')
        if not os.path.exists(dir):
            os.makedirs(dir, 0o755)
        self._dir = dir

    def __call__(self, func, items):
        # TODO: Only submit some at once? (to support infinite stream)
        for k, v in items:
            proc = slurmproc.Process(functools.partial(func, k, v),
                                     dir=os.path.join(self._dir, k), **self._kwargs)
            logger.info('submitted slurm job %s for element "%s"', proc.job_id(), k)
            self._procs[k] = proc
            self._num += 1

        try:
            while True:
                # Obtain remaining keys by job ID.
                remaining = {proc.job_id(): key for key, proc in self._procs.items()}
                if len(remaining) == 0:
                    break
                completed = slurmproc.wait_any(set(remaining.keys()), period=self._poll_period)
                for job_id in completed:
                    key = remaining[job_id]
                    logger.debug('slurm job has terminated: %s', str(self._procs[key]))
                    # Job may have failed or succeeded.
                    # It might have written a result, which could contain an error,
                    # or terminated without writing result.
                    try:
                        output = self._procs[key].output()
                    except IOError as ex:
                        logger.warning('unable to read result for process %s: %s',
                                       str(self._procs[key]), str(ex))
                        self._errors[key] = ex
                    except slurmproc.RemoteException as ex:
                        logger.warning('exception raised in process %s: %s',
                                       str(self._procs[key]), str(ex))
                        self._errors[key] = ex
                    else:
                        yield key, output
                    finally:
                        del self._procs[key]
        finally:
            self.terminate()

        if len(self._errors) > 0:
            raise RuntimeError('errors in {} of {} jobs: {}'.format(
                len(self._errors), self._num,
                ', '.join(map(_quote, sorted(self._errors.keys())))))

    def terminate(self):
        for key, proc in self._procs.items():
            logger.debug('terminate process for element "%s"', key)
            proc.terminate()


def _quote(x):
    return '"{}"'.format(x)
