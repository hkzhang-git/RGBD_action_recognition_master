import os
import cv2
import argparse
import numpy as np
from glob import glob
from functools import partial
from multiprocessing import Pool


parser = argparse.ArgumentParser(description='optical flow extraction')
parser.add_argument('--data_root', type=str, default='/home/hkzhang/Documents/sdb_a/Action_recognition_data/')
parser.add_argument('--dataset', type=str, default='N_UCLA')
parser.add_argument('--subset_id_start', type=int, default=0)
parser.add_argument('--subset_id_end', type=int, default=3)
parser.add_argument('--num_workers', type=int, default=2)

args = parser.parse_args()

source_dir = os.path.join(args.data_root, args.dataset, 'rgb_jpg_subsets')
subset_list = os.listdir(source_dir)
subset_list.sort()
print(subset_list)

optical_flow_save_dir = os.path.join(args.data_root, args.dataset, 'opti_flow_jpg')

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def truncate(flow, min=-8, max=8):
    flow[flow>max]=max
    flow[flow<min]=min
    t_flow= np.uint8((flow + min) * (255/(max-min)))
    return t_flow


def optical_flow_extraction(save_path, source_path):
    video_id = source_path.split('/')[-1]
    optical_save_path = os.path.join(save_path, video_id)
    make_if_not_exist(optical_save_path)

    frames_list = glob(source_path + '/rgb_*.jpg')
    frames_list.sort()

    data_npy = []
    for frame_dir in frames_list:
        frame = cv2.imread(frame_dir)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        data_npy.append(frame_gray)

    optical_flow = cv2.createOptFlow_DualTVL1()
    # optical_flow = cv2.DualTVL1OpticalFlow_create()

    flow_npy = []
    for frame_id in range(1, len(data_npy)):
        # flow = cv2.calcOpticalFlowFarneback(data_npy[frame_id-1], data_npy[frame_id], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = optical_flow.calc(data_npy[frame_id-1], data_npy[frame_id], None)
        t_flow = truncate(flow)

        cv2.imwrite(optical_save_path + '/opti_flow_x_' + '%05d' % frame_id + '.jpg', t_flow[..., 0])
        cv2.imwrite(optical_save_path + '/opti_flow_y_' + '%05d' % frame_id + '.jpg', t_flow[..., 1])


def subset_opti_flow(id, save_path, video_list, source_dir):
    optical_flow_extraction(save_path, os.path.join(source_dir, video_list[id]))



def main():

    for id in range(args.subset_id_start, args.subset_id_end):
        video_list = os.listdir(os.path.join(source_dir, subset_list[id]))
        pool = Pool(args.num_workers)
        partial_subset_opti_flow = partial(subset_opti_flow, save_path=optical_flow_save_dir,
                                           video_list = video_list, source_dir=os.path.join(source_dir, subset_list[id]))
        N = len(video_list)
        _ = pool.map(partial_subset_opti_flow, range(N))
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()


