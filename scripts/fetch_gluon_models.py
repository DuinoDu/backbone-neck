import os
from gluoncv.model_zoo import get_model, get_model_list

root = '/home/public_data/min.du/mxnet_models'
if not os.path.exists(root):
    os.makedirs(root)

for name in get_model_list():
    get_model(name, pretrained=True, root=root)
