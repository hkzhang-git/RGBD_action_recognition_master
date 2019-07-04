import numpy as np
import torch


def channel_shuffle(x, groups):
    if groups == 1:
        return x
    else:
        batchsize, num_channels, frame_num, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, frame_num, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, frame_num, height, width)
        return x

# groups = 3
# index = torch.from_numpy(np.arange(0, 42))
# index = index.view(3, 14)
# index = torch.transpose(index, 0, 1).contiguous()
# index = index.view(42)
index = np.arange(0, 42)
index = np.reshape(index, (3, 14))
index = np.transpose(index, (1, 0))
index = index.flatten()



