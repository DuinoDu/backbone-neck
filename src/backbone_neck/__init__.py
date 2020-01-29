# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
"""Introduce"""

from __future__ import absolute_import
import os
import logging
from .registry import build_from_cfg, BACKBONES, NECKS
from .registry import BACKBONES_MXNET, NECKS_MXNET
from . import gluon
from . import torch


__all__ = ['build_backbone', 'build_neck', 'set_backend']


__version__ = '0.0.1'


BACKEND = None
have_torch = False
have_mxnet = False
try:
    from torch import nn
    have_torch = True
except ImportError as e:
    print("torch not found.")

try:
    from mxnet.gluon import nn
    have_mxnet = True
except ImportError as e:
    print("mxnet not found.")

if hasattr(os.environ, 'BackboneNeck_Backend'):
    BACKEND = os.environ['BackboneNeck_Backend']
    assert BACKEND in ('torch', 'mxnet')
else:
    if have_mxnet:
        BACKEND = 'mxnet'
    elif have_torch:
        BACKEND = 'torch'
    else:
        raise EnvironmentError("Please install pytorch or mxnet")


def set_backend(_backend):
    global BACKEND
    assert _backend in ('torch', 'mxnet')
    BACKEND = _backend


def build_backbone(cfg, backend=BACKEND):
    if backend == 'mxnet':
        r = BACKBONES_MXNET
    elif backend == 'torch':
        r = BACKBONES
    else:
        raise ValueError("Unknown backend: %s" % BACKEND)
    logging.info("backend: %s" % backend)
    return build(cfg, r)


def build_neck(cfg, backend=BACKEND):
    if backend == 'mxnet':
        r = NECKS_MXNET
    elif backend == 'torch':
        r = NECKS
    else:
        raise ValueError("Unknown backend: %s" % BACKEND)
    logging.info("backend: %s" % backend)
    return build(cfg, r)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        ret = None
        if have_torch:
            ret = nn.Sequential(*modules)
        elif have_mxnet:
            ret = nn.Sequential()
            with ret.name_scope():
                for m in modules:
                    ret.add(m)
        else:
            pass
        return ret
    else:
        return build_from_cfg(cfg, registry, default_args)
