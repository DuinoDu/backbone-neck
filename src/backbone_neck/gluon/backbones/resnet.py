import logging
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
import gluoncv as gcv
from gluoncv.model_zoo import get_model

from ...registry import BACKBONES_MXNET


@BACKBONES_MXNET.register_module
class ResNet(HybridBlock):
    """ResNet backbone using gluoncv.

    name: 

    """
    def __init__(self, 
                 name,
                 out_indices=(0,1,2,3),
                 **kwargs):
        super(ResNet, self).__init__()
        self.out_indices = out_indices
        assert len(out_indices) <= 4
        self.ctx = getattr(kwargs, 'ctx', mx.context.cpu())
        backbone = get_model(name, **kwargs)
        # check backbone 
        for layer in ['conv1', 'bn1', 'relu', 'maxpool',
                      'layer1', 'layer2', 'layer3', 'layer4']:
            assert hasattr(backbone, layer)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.init_weights()

    def init_weights(self):
        for param in self.collect_params().values():
            if param._data is not None:
                continue
            logging.info('init %s' % param.name)
            param.initialize(default_init=mx.init.Normal(sigma=0.01), 
                             ctx=self.ctx)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = list()
        outs.append(self.layer1(x))
        outs.append(self.layer2(outs[0]))
        outs.append(self.layer3(outs[1]))
        outs.append(self.layer4(outs[2]))
        return [out for ind, out in enumerate(outs) if ind in self.out_indices]
