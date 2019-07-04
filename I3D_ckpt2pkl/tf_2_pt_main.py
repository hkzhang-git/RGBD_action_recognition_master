import torch
from torch.nn import DataParallel
from functions_for_transfer_2 import weight_extract, transfer_conv, transfer_mixed_conv, transfer_mixed_conv_5b
from functions_for_transfer_2 import make_if_not_exist, transfer_conv_depth, transfer_conv_rgbd
import sys
sys.path.append('../training/models')
from nets_i3d import I3D

pt_model_dir = './py_model/'
make_if_not_exist(pt_model_dir)

# extracted weight from checkpoint file of tf
pretrained_model_dir = './tf_weight/rgb_scratch_kin600/model.ckpt'
keys, tensors = weight_extract(pretrained_model_dir)

# ***********************************************************************************rgb data based model
net_rgb = DataParallel(I3D(num_classes=60, input_channel=3))
rgb_state_dict = net_rgb.state_dict()

# transfer
transfer_conv(rgb_state_dict, keys, tensors, 'features.0', 'Conv3d_1a_7x7')
transfer_conv(rgb_state_dict, keys, tensors, 'features.2', 'Conv3d_2b_1x1')
transfer_conv(rgb_state_dict, keys, tensors, 'features.3', 'Conv3d_2c_3x3')

transfer_mixed_conv(rgb_state_dict, keys, tensors, 'features.5', 'Mixed_3b')
transfer_mixed_conv(rgb_state_dict, keys, tensors, 'features.6', 'Mixed_3c')
transfer_mixed_conv(rgb_state_dict, keys, tensors, 'features.8', 'Mixed_4b')
transfer_mixed_conv(rgb_state_dict, keys, tensors, 'features.9', 'Mixed_4c')
transfer_mixed_conv(rgb_state_dict, keys, tensors, 'features.10', 'Mixed_4d')
transfer_mixed_conv(rgb_state_dict, keys, tensors, 'features.11', 'Mixed_4e')
transfer_mixed_conv(rgb_state_dict, keys, tensors, 'features.12', 'Mixed_4f')
transfer_mixed_conv_5b(rgb_state_dict, keys, tensors, 'features.14', 'Mixed_5b')
transfer_mixed_conv(rgb_state_dict, keys, tensors, 'features.15', 'Mixed_5c')

net_rgb.load_state_dict(rgb_state_dict)
py_model_save_dir = pt_model_dir + 'I3D_kin_rgb.pkl'
torch.save(net_rgb.state_dict(), py_model_save_dir)

# **********************************************************************************depth data based model
net_depth = DataParallel(I3D(num_classes=60, input_channel=1))
depth_state_dict = net_depth.state_dict()

# transfer
transfer_conv_depth(depth_state_dict, keys, tensors, 'features.0', 'Conv3d_1a_7x7')
transfer_conv(depth_state_dict, keys, tensors, 'features.2', 'Conv3d_2b_1x1')
transfer_conv(depth_state_dict, keys, tensors, 'features.3', 'Conv3d_2c_3x3')

transfer_mixed_conv(depth_state_dict, keys, tensors, 'features.5', 'Mixed_3b')
transfer_mixed_conv(depth_state_dict, keys, tensors, 'features.6', 'Mixed_3c')
transfer_mixed_conv(depth_state_dict, keys, tensors, 'features.8', 'Mixed_4b')
transfer_mixed_conv(depth_state_dict, keys, tensors, 'features.9', 'Mixed_4c')
transfer_mixed_conv(depth_state_dict, keys, tensors, 'features.10', 'Mixed_4d')
transfer_mixed_conv(depth_state_dict, keys, tensors, 'features.11', 'Mixed_4e')
transfer_mixed_conv(depth_state_dict, keys, tensors, 'features.12', 'Mixed_4f')
transfer_mixed_conv_5b(depth_state_dict, keys, tensors, 'features.14', 'Mixed_5b')
transfer_mixed_conv(depth_state_dict, keys, tensors, 'features.15', 'Mixed_5c')

net_depth.load_state_dict(depth_state_dict)
py_model_save_dir = pt_model_dir + 'I3D_kin_depth.pkl'
torch.save(net_depth.state_dict(), py_model_save_dir)

# *********************************************************************************rgbd data based model
net_rgbd = DataParallel(I3D(num_classes=60, input_channel=4))
rgbd_state_dict = net_rgbd.state_dict()

# transfer
transfer_conv_rgbd(rgbd_state_dict, keys, tensors, 'features.0', 'Conv3d_1a_7x7')
transfer_conv(rgbd_state_dict, keys, tensors, 'features.2', 'Conv3d_2b_1x1')
transfer_conv(rgbd_state_dict, keys, tensors, 'features.3', 'Conv3d_2c_3x3')

transfer_mixed_conv(rgbd_state_dict, keys, tensors, 'features.5', 'Mixed_3b')
transfer_mixed_conv(rgbd_state_dict, keys, tensors, 'features.6', 'Mixed_3c')
transfer_mixed_conv(rgbd_state_dict, keys, tensors, 'features.8', 'Mixed_4b')
transfer_mixed_conv(rgbd_state_dict, keys, tensors, 'features.9', 'Mixed_4c')
transfer_mixed_conv(rgbd_state_dict, keys, tensors, 'features.10', 'Mixed_4d')
transfer_mixed_conv(rgbd_state_dict, keys, tensors, 'features.11', 'Mixed_4e')
transfer_mixed_conv(rgbd_state_dict, keys, tensors, 'features.12', 'Mixed_4f')
transfer_mixed_conv_5b(rgbd_state_dict, keys, tensors, 'features.14', 'Mixed_5b')
transfer_mixed_conv(rgbd_state_dict, keys, tensors, 'features.15', 'Mixed_5c')

net_rgbd.load_state_dict(rgbd_state_dict)
py_model_save_dir = pt_model_dir + 'I3D_kin_rgbd.pkl'
torch.save(net_rgbd.state_dict(), py_model_save_dir)