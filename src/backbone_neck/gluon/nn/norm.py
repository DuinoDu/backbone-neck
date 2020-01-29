import mxnet.gluon.nn as nn
import mxnet.gluon as gluon
import gluoncv as gcv

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    'BN': ('bn', nn.BatchNorm),
    'SyncBN': ('syncbn', gluon.contrib.nn.SyncBatchNorm),
    'IN': ('in', nn.InstanceNorm),
    'LN': ('ln', nn.LayerNorm),
    'GN': ('gn', gcv.nn.GroupNorm),
    'SN': ('sn', None),
}


def build_norm_layer(cfg):
    """ Build normalization layer

    Parameters
    ----------
    cfg (dict): cfg should contain:
        type (str): identify norm layer type.
        layer args: args needed to instantiate a norm layer.
        requires_grad (bool): [optional] whether stop gradient updates

    Returns
    -------
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError
        layer = norm_layer(**cfg_)

    requires_grad = cfg_.pop('requires_grad', True)

    # cfg_.setdefault('eps', 1e-5)
    # if layer_type != 'GN':
    #     layer = norm_layer(num_features, **cfg_)
    #     if layer_type == 'SyncBN':
    #         layer._specify_ddp_gpu_num(1)
    # else:
    #     assert 'num_groups' in cfg_
    #     layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.collect_params().values():
        param.grad_req = 'null'

    return layer
