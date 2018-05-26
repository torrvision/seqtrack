import functools

import logging
logger = logging.getLogger(__name__)

import slurmproc


class SlurmDictMapper(object):

    def __init__(self, **kwargs):
        '''
        Args:
            kwargs: For slurmproc.Process.
        '''
        self._kwargs = kwargs

    def __call__(self, func, items):
        procs = {}
        errors = {}
        # TODO: Only submit some at once? (to support infinite stream)
        for k, v in items:
            proc = slurmproc.Process(functools.partial(func, k, v), **self._kwargs)
            logger.info('started slurm job %s for element "%s"', str(proc._job_id), k)
            procs[k] = proc
        # TODO: Return in order that they finish?
        # TODO: Cancel jobs on interrupt.
        for k, proc in procs.items():
            try:
                result = proc.wait()
            except Exception as ex:
                logger.warning('error for element "%s": %s', k, str(ex))
                errors[k] = ex
                continue
            yield k, result
        if len(errors) > 0:
            raise RuntimeError('errors in {} of {} jobs'.format(len(errors), len(procs)))
