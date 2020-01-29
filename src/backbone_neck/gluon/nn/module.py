import warnings

import mxnet as mx
from .conv import build_conv_layer
from .norm import build_norm_layer
from .activation import build_act_layer


__all__ = ['ConvModule']


class ConvModule(mx.gluon.HybridBlock):
    """A conv block that contains conv/norm/activation layers.
    
    Parameters
    ----------
    in_channels (int):
        Same as nn.Conv2d.
    channels (int):
        Same as nn.Conv2d.
    kernel_size (int or tuple[int]):
        Same as nn.Conv2d.
    stride (int or tuple[int]):
        Same as nn.Conv2d.
    padding (int or tuple[int]):
        Same as nn.Conv2d.
    dilation (int or tuple[int]):
        Same as nn.Conv2d.
    groups (int):
        Same as nn.Conv2d.
    bias (bool or str): If specified as `auto`, it will be decided by the
        norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
        False.
    conv_cfg (dict):
        Config dict for convolution layer.
    norm_cfg (dict):
        Config dict for normalization layer.
    act_cfg (dict):
        Config dict for activation layer.
    activation (str or None):
        Activation type, "ReLU" by default. [deprecated, using act_cfg]
    order (tuple[str]):
        The order of conv/norm/activation layers. It is a
        sequence of "conv", "norm" and "act". Examples are
        ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(self,
                 channels=None,
                 kernel_size=None,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 in_channels=0,
                 activation=None,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None or activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias:
            bias = not self.with_norm
        self.with_bias = bias
        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        if activation:
            assert act_cfg is None
            act_cfg = dict(
                type='Activation',
                activation=activation)
        if conv_cfg is None:
            assert channels is not None
            assert kernel_size is not None
            conv_cfg = dict(
                type='Conv',
                channels=channels,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=bias,
                activation=None,
                in_channels=in_channels
            )
        else:
            if channels is not None or kernel_size is not None:
                warnings.warn('using conv_cfg, ignore channels, kernel_size')

        with self.name_scope():
            # build normalization layers
            if self.with_norm:
                self.norm = build_norm_layer(norm_cfg)
            # build activation layers
            if self.with_activation:
                self.act = build_act_layer(act_cfg)
            # build convolution layer
            self.conv = build_conv_layer(conv_cfg)
        # # export the attributes of self.conv to a higher level for convenience
        # self.channels = self.conv.channels
        # self.kernel_size = self.conv.kernel_size
        # self.stride = self.conv.strides
        # self.padding = self.conv.padding
        # self.dilation = self.conv.dilation
        # self.transposed = self.conv.transposed
        # self.output_padding = self.conv.output_padding
        # self.groups = self.conv.groups
        # self.in_channels = self.conv.in_channels

        ## Use msra init by default
        #self.init_weights()

    def init_weights(self):
        from mmcv.cnn import constant_init, kaiming_init
        nonlinearity = 'relu' if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.act(x)
        return x
