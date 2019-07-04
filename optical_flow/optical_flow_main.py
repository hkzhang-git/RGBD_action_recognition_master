import os
import argparse
from glob import glob
from functools import partial
from multiprocessing import Pool
from optical_flow.functions_for_optical_flow import subset_opti_flow


parser = argparse.ArgumentParser(description='optical flow extraction')
parser.add_argument('--data_root', type=str, default='/home/hkzhang/Documents/sdb_a/Action_recognition_data/')
parser.add_argument('--dataset', type=str, default='NTU')
parser.add_argument('--set_size', type=int, default=[256, 310])
parser.add_argument('--subset_id_start', type=int, default=0)
parser.add_argument('--subset_id_end', type=int, default=3)
parser.add_argument('--num_workers', type=int, default=2)

args = parser.parse_args()

source_dir = os.path.join(args.data_root, args.dataset, 'rgb')
subset_list = os.listdir(source_dir)
subset_list.sort()
print(subset_list)

optical_flow_save_dir = os.path.join(args.data_root, args.dataset, 'optical_flow')


def main():

    for id in range(args.subset_id_start, args.subset_id_end):
        video_list = glob(os.path.join(source_dir, subset_list[id]) + '/*_rgb.avi')

        pool = Pool(args.num_workers)
        partial_subset_opti_flow = partial(subset_opti_flow, save_path=optical_flow_save_dir,
                                           video_list = video_list,
                                           set_size = args.set_size)
        N = len(video_list)
        _ = pool.map(partial_subset_opti_flow, range(N))
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()


