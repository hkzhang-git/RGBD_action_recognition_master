import os
import torch
import numpy as np
from tensorflow.python import pywrap_tensorflow

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def weight_extract(file_dir):
    reader = pywrap_tensorflow.NewCheckpointReader(file_dir)
    var_to_shape_map = reader.get_variable_to_shape_map()
    keys_list=[]
    tensors=[]

    for key in sorted(var_to_shape_map):
        keys_list.append(key)
        tensors.append(reader.get_tensor(key))

    return keys_list, tensors


def get_value(keys, tensors, layer_key):
    id = np.where(keys == layer_key)[0][0]
    data = tensors[id]

    if layer_key.split('/')[-1] in ['beta', 'moving_mean', 'moving_variance']:
        return torch.from_numpy(data).permute(4, 3, 0, 1, 2).squeeze()
    else:
        return data.shape[-1], torch.from_numpy(data).permute(4, 3, 0, 1, 2)


# transfer_conv(state_dict, keys, tensors, 'features.0', 'Conv2d_1a_7x7')
def transfer_conv(state_dict, keys, tensors, layer_key_pt, layer_key_tf):
    keys = np.array(keys)
    key_tf = layer_key_tf
    key_pt = 'module.' + layer_key_pt

    # conv
    key_tf_weight = key_tf + '/conv_3d/w'
    key_pt_weight = key_pt + '.conv.weight'
    oup, state_dict[key_pt_weight] = get_value(keys, tensors, key_tf_weight)
    # bn_gamma
    key_tf_bn_weight = torch.ones(oup)
    key_pt_bn_weight = key_pt + '.bn.weight'
    state_dict[key_pt_bn_weight] = key_tf_bn_weight
    # bn_beta
    key_tf_bn_bias = key_tf + '/batch_norm/beta'
    key_pt_bn_bias = key_pt + '.bn.bias'
    state_dict[key_pt_bn_bias] = get_value(keys, tensors, key_tf_bn_bias)
    # bn_mean
    key_tf_bn_mean = key_tf + '/batch_norm/moving_mean'
    key_pt_bn_mean = key_pt + '.bn.running_mean'
    state_dict[key_pt_bn_mean] = get_value(keys, tensors, key_tf_bn_mean)
    # bn_var
    key_tf_bn_var = key_tf + '/batch_norm/moving_variance'
    key_pt_bn_var = key_pt + '.bn.running_var'
    state_dict[key_pt_bn_var] = get_value(keys, tensors, key_tf_bn_var)


def get_value_depth(keys, tensors, layer_key):
    id = np.where(keys == layer_key)[0][0]
    data = tensors[id]
    weight = torch.from_numpy(data).permute(4, 3, 0, 1, 2)

    return data.shape[-1], torch.sum(weight, 1, keepdim=True)


def transfer_conv_depth(state_dict, keys, tensors, layer_key_pt, layer_key_tf):
    keys = np.array(keys)
    key_tf = layer_key_tf
    key_pt = 'module.' + layer_key_pt

    # conv
    key_tf_weight = key_tf + '/conv_3d/w'
    key_pt_weight = key_pt + '.conv.weight'
    oup, state_dict[key_pt_weight] = get_value_depth(keys, tensors, key_tf_weight)
    # bn_gamma
    key_tf_bn_weight = torch.ones(oup)
    key_pt_bn_weight = key_pt + '.bn.weight'
    state_dict[key_pt_bn_weight] = key_tf_bn_weight
    # bn_beta
    key_tf_bn_bias = key_tf + '/batch_norm/beta'
    key_pt_bn_bias = key_pt + '.bn.bias'
    state_dict[key_pt_bn_bias] = get_value(keys, tensors, key_tf_bn_bias)
    # bn_mean
    key_tf_bn_mean = key_tf + '/batch_norm/moving_mean'
    key_pt_bn_mean = key_pt + '.bn.running_mean'
    state_dict[key_pt_bn_mean] = get_value(keys, tensors, key_tf_bn_mean)
    # bn_var
    key_tf_bn_var = key_tf + '/batch_norm/moving_variance'
    key_pt_bn_var = key_pt + '.bn.running_var'
    state_dict[key_pt_bn_var] = get_value(keys, tensors, key_tf_bn_var)


def get_value_rgbd(keys, tensors, layer_key):
    id = np.where(keys == layer_key)[0][0]
    data = tensors[id]
    rgb_weight = torch.from_numpy(data).permute(4, 3, 0, 1, 2)
    depth_weight = torch.sum(rgb_weight, 1, keepdim=True)

    return data.shape[-1], torch.cat((depth_weight/2, rgb_weight/2), 1)


def transfer_conv_rgbd(state_dict, keys, tensors, layer_key_pt, layer_key_tf):
    keys = np.array(keys)
    key_tf = layer_key_tf
    key_pt = 'module.' + layer_key_pt

    # conv
    key_tf_weight = key_tf + '/conv_3d/w'
    key_pt_weight = key_pt + '.conv.weight'
    oup, state_dict[key_pt_weight] = get_value_rgbd(keys, tensors, key_tf_weight)
    # bn_gamma
    key_tf_bn_weight = torch.ones(oup)
    key_pt_bn_weight = key_pt + '.bn.weight'
    state_dict[key_pt_bn_weight] = key_tf_bn_weight
    # bn_beta
    key_tf_bn_bias = key_tf + '/batch_norm/beta'
    key_pt_bn_bias = key_pt + '.bn.bias'
    state_dict[key_pt_bn_bias] = get_value(keys, tensors, key_tf_bn_bias)
    # bn_mean
    key_tf_bn_mean = key_tf + '/batch_norm/moving_mean'
    key_pt_bn_mean = key_pt + '.bn.running_mean'
    state_dict[key_pt_bn_mean] = get_value(keys, tensors, key_tf_bn_mean)
    # bn_var
    key_tf_bn_var = key_tf + '/batch_norm/moving_variance'
    key_pt_bn_var = key_pt + '.bn.running_var'
    state_dict[key_pt_bn_var] = get_value(keys, tensors, key_tf_bn_var)


# transfer_mixed_conv(state_dict, keys, tensors, 'features.5', 'Mixed_3b')
def transfer_mixed_conv(state_dict, keys, tensors, layer_key_pt, layer_key_tf):
    # branch 0
    key_tf = layer_key_tf + '/Branch_0/Conv3d_0a_1x1'
    key_pt = layer_key_pt + '.branch0.0'
    transfer_conv(state_dict, keys, tensors, key_pt, key_tf)
    # branch 1.0
    key_tf = layer_key_tf + '/Branch_1/Conv3d_0a_1x1'
    key_pt = layer_key_pt + '.branch1.0'
    transfer_conv(state_dict, keys, tensors, key_pt, key_tf)
    # branch 1.1
    key_tf = layer_key_tf + '/Branch_1/Conv3d_0b_3x3'
    key_pt = layer_key_pt + '.branch1.1'
    transfer_conv(state_dict, keys, tensors, key_pt, key_tf)
    # branch 2.0
    key_tf = layer_key_tf + '/Branch_2/Conv3d_0a_1x1'
    key_pt = layer_key_pt + '.branch2.0'
    transfer_conv(state_dict, keys, tensors, key_pt, key_tf)
    # branch 2.1
    key_tf = layer_key_tf + '/Branch_2/Conv3d_0b_3x3'
    key_pt = layer_key_pt + '.branch2.1'
    transfer_conv(state_dict, keys, tensors, key_pt, key_tf)
    # branch 3
    key_tf = layer_key_tf + '/Branch_3/Conv3d_0b_1x1'
    key_pt = layer_key_pt + '.branch3.1'
    transfer_conv(state_dict, keys, tensors, key_pt, key_tf)


def transfer_mixed_conv_5b(state_dict, keys, tensors, layer_key_pt, layer_key_tf):
    # branch 0
    key_tf = layer_key_tf + '/Branch_0/Conv3d_0a_1x1'
    key_pt = layer_key_pt + '.branch0.0'
    transfer_conv(state_dict, keys, tensors, key_pt, key_tf)
    # branch 1.0
    key_tf = layer_key_tf + '/Branch_1/Conv3d_0a_1x1'
    key_pt = layer_key_pt + '.branch1.0'
    transfer_conv(state_dict, keys, tensors, key_pt, key_tf)
    # branch 1.1
    key_tf = layer_key_tf + '/Branch_1/Conv3d_0b_3x3'
    key_pt = layer_key_pt + '.branch1.1'
    transfer_conv(state_dict, keys, tensors, key_pt, key_tf)
    # branch 2.0
    key_tf = layer_key_tf + '/Branch_2/Conv3d_0a_1x1'
    key_pt = layer_key_pt + '.branch2.0'
    transfer_conv(state_dict, keys, tensors, key_pt, key_tf)
    # branch 2.1************************************0a_3x3
    key_tf = layer_key_tf + '/Branch_2/Conv3d_0a_3x3'
    key_pt = layer_key_pt + '.branch2.1'
    transfer_conv(state_dict, keys, tensors, key_pt, key_tf)
    # branch 3
    key_tf = layer_key_tf + '/Branch_3/Conv3d_0b_1x1'
    key_pt = layer_key_pt + '.branch3.1'
    transfer_conv(state_dict, keys, tensors, key_pt, key_tf)


