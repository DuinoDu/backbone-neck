from mxnet.gluon import nn

act_cfg = {
    'Activation': nn.Activation,
    'LeakyReLU': nn.LeakyReLU,
    'PReLU': nn.PReLU,
    'ELU': nn.ELU,
    'SELU': nn.SELU,
    'Swish': nn.Swish,
    'GELU': nn.GELU,
}


def build_act_layer(cfg, *args, **kwargs):
    """ Build activation layer

    Parameters
    ----------
    cfg : dict or None
        cfg should contain:
            type (str): identify act layer type.
            layer args: args needed to instantiate a act layer.

    Return
    ------
    layer (nn.Module): created act layer
    """
    if cfg is None:
        cfg_ = dict(type='Activation', activation='relu')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in act_cfg:
        raise KeyError('Unrecognized act type {}'.format(layer_type))
    else:
        act_layer = act_cfg[layer_type]

    layer = act_layer(*args, **kwargs, **cfg_)
    return layer
