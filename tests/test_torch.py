# -*- coding: utf-8 -*-
from .context import backbone_neck
import unittest
import sys

backbone_neck.set_backend('torch')


def _input(h=128, w=128):
    try:
        import torch
    except ImportError as e:
        print("torch not installed, exit")
        sys.exit()
    inputs = torch.rand(1, 3, h, w)
    return inputs


def _features(h=128, w=128):
    cfg = dict(type='ResNet', depth=50)
    backbone = backbone_neck.build_backbone(cfg, backend='torch')
    features = backbone.forward(_input(h, w))
    return features


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_build_backbone(self):
        cfg = dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0,1,2,3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            style='caffe')
        backbone = backbone_neck.build_backbone(cfg, backend='torch')
        backbone.eval()
        level_outputs = backbone.forward(_input(128, 128))
        # 32x32, 16x16, 8x8, 4x4
        for ind, l in enumerate(level_outputs):
            s_shape = tuple(l.shape)
            self.assertEqual(s_shape[3], (128 / 4 / (2**ind)))

    def test_build_neck(self):
        cfg = dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            start_level=1,
            add_extra_convs=True,
            extra_convs_on_inputs=False,
            relu_before_extra_convs=True)
        neck = backbone_neck.build_neck(cfg, backend='torch')
        level_outputs = neck.forward(_features(h=128, w=128))
        for ind, l in enumerate(level_outputs):
            s_shape = tuple(l.shape)
            self.assertEqual(s_shape[3], (128 / 8 / (2**ind)))


if __name__ == '__main__':
    unittest.main()
