# -*- coding: utf-8 -*-

from .context import backbone_neck
from backbone_neck.gluon.nn import activation
import unittest
import sys
import numpy as np


def _input(h=128, w=128):
    try:
        import mxnet.ndarray as nd
    except ImportError as e:
        print('mxnet not install, exit')
        sys.exit()
    inputs = nd.random.randn(1, 3, h, w)
    return inputs


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_activation(self):
        cfg = dict(
            type='Activation',
            activation='relu')
        relu = activation.build_act_layer(cfg)
        x = _input(128, 128)
        y = relu(x)
        x, y = x.asnumpy(), y.asnumpy()
        self.assertTrue(np.sum(y[x < 0]) == 0)
        self.assertTrue(np.alltrue(y[x > 0] == x[x > 0]))

    def test_LeakyReLU(self):
        cfg = dict(
            type='LeakyReLU',
            alpha=0.5)
        relu = activation.build_act_layer(cfg)
        x = _input(128, 128)
        y = relu(x)
        x, y = x.asnumpy(), y.asnumpy()
        self.assertTrue(np.sum(y[x < 0]) < 0)
        self.assertTrue(np.alltrue(y[x > 0] == x[x > 0]))


if __name__ == '__main__':
    unittest.main()
