from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import itertools
import functools
import os
import subprocess
import tempfile

import logging
logger = logging.getLogger(__name__)

from seqtrack import helpers
import slurmproc


class SlurmDictMapper(object):

    def __init__(self, poll_period=1, dir=None, tempdir=None, max_submit=None, **kwargs):
        '''
        Args:
            dir: If none, then create a new directory under tempdir.
            tempdir: Only used if dir is none.
            kwargs: For slurmproc.Process.
        '''
        self._poll_period = poll_period
        self._max_submit = max_submit
        self._kwargs = kwargs

        self._procs = {}  # Submitted and incomplete jobs. Dict that maps key to Process.
        self._errors = {}  # Dict that maps key to Exception.
        self._num = 0  # Number of elements read from input stream.

        if not dir:
            assert tempdir
            if not os.path.exists(tempdir):
                os.makedirs(tempdir, 0o755)
            timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            dir = tempfile.mkdtemp(dir=tempdir, prefix=('tmp_{}_'.format(timestamp)))
        else:
            if os.path.exists(dir):
                raise RuntimeError('dir already exists')
        if not os.path.exists(dir):
            os.makedirs(dir, 0o755)
        self._dir = dir

    def __call__(self, func, items):
        try:
            item_stream = iter(items)
            items_remaining = True
            while items_remaining or self._procs:
                if items_remaining and _below_limit(len(self._procs), self._max_submit):
                    # We can submit another job.
                    try:
                        key_value = next(item_stream)
                    except StopIteration:
                        items_remaining = False
                    else:
                        k, v = key_value
                        self._submit(func, k, v)
                        self._num += 1
                else:
                    for key, output in self._wait_any():
                        yield key, output
        finally:
            self.terminate()

        if len(self._errors) > 0:
            raise RuntimeError('errors in {} of {} jobs: {}'.format(
                len(self._errors), self._num,
                ', '.join(map(_quote, sorted(self._errors.keys())))))

    def _submit(self, func, key, value):
        try:
            proc = slurmproc.Process(functools.partial(func, MapContext(key), value),
                                     dir=os.path.join(self._dir, key), job_name=str(key),
                                     **self._kwargs)
        except subprocess.CalledProcessError as ex:
            logger.warning('could not submit process %s: %s', str(key), str(ex))
            self._errors[key] = ex
        else:
            logger.info('submitted slurm job %s for element "%s"', proc.job_id(), key)
            self._procs[key] = proc

    def _wait_any(self):
        '''Returns iterator for results of completed jobs.'''
        # Wait for a job to finish.
        by_job_id = {proc.job_id(): key for key, proc in self._procs.items()}
        completed = slurmproc.wait_any(set(by_job_id.keys()), period=self._poll_period)
        for job_id in completed:
            key = by_job_id[job_id]
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

    def terminate(self):
        for key, proc in self._procs.items():
            logger.debug('terminate process for element "%s"', key)
            proc.terminate()


# class SlurmDictGroupMapper(object):
# 
#     def __init__(self, poll_period=1, dir=None, tempdir=None, group_size=1, **kwargs):
#         '''
#         Args:
#             dir: If none, then create a new directory under tempdir.
#             tempdir: Only used if dir is none.
#             kwargs: For slurmproc.Process.
#         '''
#         self._poll_period = poll_period
#         self._group_size = group_size
#         self._kwargs = kwargs
# 
#         self._procs = {}  # Incomplete jobs. Dict that maps job_id to Process.
#         self._group_keys = {}  # Dict that maps job_id to list of keys.
#         self._errors = {}  # Dict that maps job_id to Exception.
#         self._num_items = 0
#         self._num_jobs = 0
# 
#         if not dir:
#             assert tempdir
#             if not os.path.exists(tempdir):
#                 os.makedirs(tempdir, 0o755)
#             timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
#             dir = tempfile.mkdtemp(dir=tempdir, prefix=('tmp_{}_'.format(timestamp)))
#         else:
#             if os.path.exists(dir):
#                 raise RuntimeError('dir already exists')
#         if not os.path.exists(dir):
#             os.makedirs(dir, 0o755)
#         self._dir = dir
# 
#     def __call__(self, func, items):
#         # TODO: Only submit some at once? (to support infinite stream)
#         while True:
#             assert self._group_size >= 1
#             group_items = list(itertools.islice(items, self._group_size))
#             if len(group_items) == 0:  # Reached end of list.
#                 break
#             group_keys = [k for k, v in group_items]
#             print('group_keys:', group_keys)
#             group_name = '_'.join(sorted(group_keys))
#             proc = slurmproc.Process(functools.partial(helpers.map_dict_list, func, group_items),
#                                      dir=os.path.join(self._dir, group_name), **self._kwargs)
#             logger.info('submitted slurm job %s for elements %s', proc.job_id(), group_keys)
#             self._group_keys[proc.job_id()] = set(group_keys)
#             self._procs[proc.job_id()] = proc
#             self._num_items += len(group_items)
#             self._num_jobs += 1
# 
#         try:
#             while True:
#                 # Obtain remaining keys by job ID.
#                 # remaining = {proc.job_id(): key for key, proc in self._procs.items()}
#                 if len(self._procs) == 0:
#                     break
#                 completed = slurmproc.wait_any(set(self._procs.keys()), period=self._poll_period)
#                 for job_id in completed:
#                     keys = self._group_keys[job_id]
#                     logger.debug('slurm job has terminated: %s', str(self._procs[job_id]))
#                     # Job may have failed or succeeded.
#                     # It might have written a result, which could contain an error,
#                     # or terminated without writing result.
#                     try:
#                         # Get result of map_dict.
#                         output_items = self._procs[job_id].output()
#                     except IOError as ex:
#                         # TODO: How to get partial results if job is killed?
#                         logger.warning('unable to read result for process %s: %s',
#                                        str(self._procs[job_id]), str(ex))
#                         self._errors[job_id] = ex
#                     except slurmproc.RemoteException as ex:
#                         # TODO: How to avoid losing all results here?
#                         logger.warning('exception raised in process %s: %s',
#                                        str(self._procs[job_id]), str(ex))
#                         self._errors[job_id] = ex
#                     else:
#                         for k, v in output_items:
#                             yield k, v
#                     finally:
#                         del self._procs[job_id]
#                         del self._group_keys[job_id]
#         finally:
#             self.terminate()
# 
#         if len(self._errors) > 0:
#             raise RuntimeError('errors in {} of {} jobs: {}'.format(
#                 len(self._errors), self._num_jobs, sorted(self._errors.keys())))
# 
#     def terminate(self):
#         for job_id, proc in self._procs.items():
#             logger.debug('terminate process for elements %s', self._group_keys[job_id])
#             proc.terminate()


def _quote(x):
    return '"{}"'.format(x)


class MapContext(object):

    def __init__(self, name):
        self.name = name

    def tmp_dir(self):
        return get_tmp_dir()


def get_tmp_dir():
    return '/raid/local_scratch/{}-{}'.format(os.environ['SLURM_JOB_USER'],
                                              os.environ['SLURM_JOB_ID'])


def _is_slurm_job():
    return 'SLURM_JOB_ID' in os.environ


def _below_limit(a, b):
    return b is None or b <= 0 or a < b
