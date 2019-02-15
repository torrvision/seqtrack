from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import siamfc
from . import regress
from . import siamflow


BY_NAME = {
    'siamfc': siamfc.SiamFC,
    'regress': regress.MotionRegressor,
    'siamflow': siamflow.SiamFlow,
}
