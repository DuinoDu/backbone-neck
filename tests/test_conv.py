# -*- coding: utf-8 -*-

from .context import backbone_neck
import unittest
import sys
from backbone_neck.gluon.nn import conv
from backbone_neck.gluon.nn import ConvModule
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


class BasicConvSuite(unittest.TestCase):
    """Basic test cases."""

    def test_conv(self):
        cfg = dict(
            type='Conv',
            channels=32,
            kernel_size=3,
            padding=1,
            use_bias=False)
        net = backbone_neck.gluon.nn.conv.build_conv_layer(cfg)
        net.initialize(ctx=[mx.cpu(0)])
        x = _input(128, 128)
        y = net(x)
        x, y = x.asnumpy(), y.asnumpy()
        self.assertEqual(y.shape[0], x.shape[0])
        self.assertEqual(y.shape[1], 32)
        self.assertEqual(y.shape[2], x.shape[2])
        self.assertEqual(y.shape[3], x.shape[3])

    def test_conv_deform(self):
        cfg = dict(
            type='DCN',
            channels=32,
            kernel_size=3,
            padding=1,
            use_bias=False)
        net = backbone_neck.gluon.nn.conv.build_conv_layer(cfg)
        net.initialize(ctx=[mx.cpu(0)])
        x = _input(128, 128)
        y = net(x)
        x, y = x.asnumpy(), y.asnumpy()
        self.assertEqual(y.shape[0], x.shape[0])
        self.assertEqual(y.shape[1], 32)
        self.assertEqual(y.shape[2], x.shape[2])
        self.assertEqual(y.shape[3], x.shape[3])

    def test_conv_oct(self):
        cfg = dict(
            type='OctConv',
            channels=32,
            kernel_size=3,
            padding=1,
            use_bias=False)
        with self.assertRaises(NotImplementedError):
            net = backbone_neck.gluon.nn.conv.build_conv_layer(cfg)


class BasicModuleSuite(unittest.TestCase):
    """Basic test cases."""

    def test_conv(self):
        conv_cfg = dict(
            type='Conv',
            channels=32,
            kernel_size=3,
            padding=1,
            use_bias=False)
        norm_cfg = dict(
            type='BN')
        act_cfg = dict(
            type='Activation',
            activation='relu')

        order = ('conv', 'norm', 'act')
        net = ConvModule(
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=order)
        net.initialize(ctx=[mx.cpu(0)])

        x = _input(128, 128)
        y = net(x)
        x, y = x.asnumpy(), y.asnumpy()
        self.assertEqual(y.shape[0], x.shape[0])
        self.assertEqual(y.shape[1], 32)
        self.assertEqual(y.shape[2], x.shape[2])
        self.assertEqual(y.shape[3], x.shape[3])


if __name__ == '__main__':
    unittest.main()
