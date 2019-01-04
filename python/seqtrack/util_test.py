from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib


def try_sub_test(test_case, **kwargs):
    try:
        context = test_case.subTest(**kwargs)
    except AttributeError:
        context = null_context()
    return context


@contextlib.contextmanager
def null_context():
    yield
