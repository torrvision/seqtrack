import tensorflow as tf

import geom

class CropTest(tf.test.TestCase):

    def testInverse(self):
        with self.test_session():
            a_b = geom.make_rect([0.1, 0.2], [0.4, 0.3])
            b_c = geom.make_rect([0.3, 0.4], [0.9, 0.8])
            c_b = geom.crop_inverse(b_c)
            a_c = geom.crop_rect(a_b, c_b)
            self.assertAllClose(a_b.eval(), geom.crop_rect(a_c, b_c).eval())
