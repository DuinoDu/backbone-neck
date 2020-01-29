import mxnet.gluon as gluon

conv_cfg = {
    'Conv': gluon.nn.Conv2D,
    'DCN': gluon.contrib.cnn.DeformableConvolution,
    'DCNv2': None,
    'OctConv': None,
}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Parameters
    ----------
    cfg : dict or None
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Return
    ------
    layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]
    if conv_layer is None:
        raise NotImplementedError

    layer = conv_layer(*args, **kwargs, **cfg_)
    return layer

