# -*- coding: utf-8 -*-

from .context import backbone_neck
import unittest
import sys
from backbone_neck.gluon.nn import norm
import numpy as np
import mxnet as mx


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

    def test_bn(self):
        cfg = dict(
            type='BN',
            use_global_stats=True,
            gamma_initializer=mx.initializer.Constant(2),
            )
        net = norm.build_norm_layer(cfg)
        net.initialize(ctx=[mx.cpu(0)])

        x = _input(128, 128)
        y = net(x)
        x, y = x.asnumpy(), y.asnumpy()
        self.assertEqual(int(x.sum()*2), int(y.sum()))


if __name__ == '__main__':
    unittest.main()
