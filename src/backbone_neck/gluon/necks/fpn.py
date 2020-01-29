import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock

from ..nn import ConvModule
from ...registry import NECKS_MXNET


@NECKS_MXNET.register_module
class FPN(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None, 
                 **kwargs):
        super(FPN, self).__init__(**kwargs)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        # create fpn
        self.lateral_convs = nn.HybridSequential()
        self.fpn_convs = nn.HybridSequential()
        with self.name_scope():
            for i in range(self.start_level, self.backbone_end_level):
                l_conv = ConvModule(
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    activation=self.activation,
                    in_channels=in_channels[i])
                fpn_conv = ConvModule(
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    in_channels=out_channels)
                self.lateral_convs.add(l_conv)
                self.fpn_convs.add(fpn_conv)
            # add extra conv layers (e.g., RetinaNet)
            extra_levels = num_outs - self.backbone_end_level + self.start_level
            if add_extra_convs and extra_levels >= 1:
                for i in range(extra_levels):
                    if i == 0 and self.extra_convs_on_inputs:
                        in_channels = self.in_channels[self.backbone_end_level - 1]
                    else:
                        in_channels = out_channels
                    extra_fpn_conv = ConvModule(
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=self.activation,
                        in_channels=in_channels)
                    self.fpn_convs.add(extra_fpn_conv)

    def hybrid_forward(self, F, *x):
        assert len(x) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(x[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.UpSampling(
                laterals[i], scale=2, sample_type='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = x[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
