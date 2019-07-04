import torch
from torch.nn import DataParallel
from nets import I3D
from functions_for_transfer import *


# extracted weight from checkpoint file of tf
pretrained_model_dir = './tf_weight/RGB_1/model.ckpt-5000'
py_model_save_dir = './py_model/rgb_kin_ucf_model.pkl'
keys, tensors = weight_extract(pretrained_model_dir)


# initial py model
net = DataParallel(I3D(num_classes=101))
state_dict = net.state_dict()

# transfer
transfer_conv(state_dict, keys, tensors, 'features.0', 'Conv2d_1a_7x7')
transfer_conv(state_dict, keys, tensors, 'features.2', 'Conv2d_2b_1x1')
transfer_conv(state_dict, keys, tensors, 'features.3', 'Conv2d_2c_3x3')

transfer_mixed_conv(state_dict, keys, tensors, 'features.5', 'Mixed_3b')
transfer_mixed_conv(state_dict, keys, tensors, 'features.6', 'Mixed_3c')
transfer_mixed_conv(state_dict, keys, tensors, 'features.8', 'Mixed_4b')
transfer_mixed_conv(state_dict, keys, tensors, 'features.9', 'Mixed_4c')
transfer_mixed_conv(state_dict, keys, tensors, 'features.10', 'Mixed_4d')
transfer_mixed_conv(state_dict, keys, tensors, 'features.11', 'Mixed_4e')
transfer_mixed_conv(state_dict, keys, tensors, 'features.12', 'Mixed_4f')
transfer_mixed_conv_5b(state_dict, keys, tensors, 'features.14', 'Mixed_5b')
transfer_mixed_conv(state_dict, keys, tensors, 'features.15', 'Mixed_5c')

net.load_state_dict(state_dict)
torch.save(net.state_dict(), py_model_save_dir)



