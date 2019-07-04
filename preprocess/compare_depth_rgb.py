import numpy as np


depth_dir = '../preprocess/data_list/NTU_2/NTU_2_depth.txt'
rgb_dir = '../preprocess/data_list/NTU_2/NTU_2_rgb.txt'

depth_info = open(depth_dir).readlines()
rgb_info = open(rgb_dir).readlines()

depth_ids = [f.split('sample:')[1].split()[0] for f in depth_info]
rgb_ids = [f.split('sample:')[1].split()[0] for f in rgb_info]

depth_ids = np.sort(np.array(depth_ids))
rgb_ids = np.sort(np.array(rgb_ids))

print('done')