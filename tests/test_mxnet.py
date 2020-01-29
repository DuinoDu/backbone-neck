# -*- coding: utf-8 -*-

from .context import backbone_neck
import unittest
import sys
import mxnet as mx


def _input(h=128, w=128):
    try:
        import mxnet.ndarray as nd
    except ImportError as e:
        print('mxnet not install, exit')
        sys.exit()
    inputs = nd.random.randn(1, 3, h, w)
    return inputs


def _features(h=128, w=128):
    cfg = dict(type='ResNet', name='resnet18_v1b')
    backbone = backbone_neck.build_backbone(cfg, backend='mxnet')
    features = backbone.forward(_input(h, w))
    return features


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_build_backbone(self):
        cfg = dict(
            type='ResNet',
            name='resnet18_v1b',
            pretrained=False)
        backbone = backbone_neck.build_backbone(cfg, backend='mxnet')
        level_outputs = backbone(_input(128, 128))
        # 32x32, 16x16, 8x8, 4x4
        for ind, l in enumerate(level_outputs):
            s_shape = tuple(l.shape)
            self.assertEqual(s_shape[3], (128 / 4 / (2**ind)))

    def test_build_neck(self):
        cfg = dict(
            type='FPN',
            in_channels=[64, 128, 256, 512],
            out_channels=256,
            num_outs=5,
            start_level=1,
            add_extra_convs=True,
            extra_convs_on_inputs=False,
            relu_before_extra_convs=True)

        neck = backbone_neck.build_neck(cfg, backend='mxnet')
        neck.initialize(ctx=[mx.cpu(0)])

        features = _features(h=128, w=128)
        level_outputs = neck(*features)

        for ind, l in enumerate(level_outputs):
            s_shape = tuple(l.shape)
            self.assertEqual(s_shape[3], (128 / 8 / (2**ind)))


if __name__ == '__main__':
    unittest.main()
