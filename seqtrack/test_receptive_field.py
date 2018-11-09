from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from seqtrack import receptive_field


class TestReceptiveField(tf.test.TestCase):

    def test_compose(self):
        # Consider z = f(y), y = g(z).
        f = receptive_field.ReceptiveField(size=(3, 3), stride=(2, 2), padding=(1, 1))
        g = receptive_field.ReceptiveField(size=(5, 5), stride=(4, 4), padding=(2, 2))
        fg = receptive_field.compose(g, f)
        self.assertAllEqual(fg.stride, (8, 8), 'strides are not equal')
        # 1 pixel in z corresponds to 3 pixels in y.
        # 1 pixel in y corresponds to 5 pixels in x.
        # 3 pixels in y corresponds to 5 + (3 - 1) * 4 = 13 pixels in x.
        self.assertAllEqual(fg.size, (13, 13), 'sizes are not equal')
        # Padding should be floor of half of size.
        self.assertAllEqual(fg.padding, (6, 6), 'padding is not equal')
