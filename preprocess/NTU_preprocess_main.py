import os
import argparse
from NTU_preprocess import samples_extraction, create_samples_list

parser = argparse.ArgumentParser(description='RGB-D samples preprocess')
parser.add_argument('--data_root', type=str, default='/home/hkzhang/Documents/sdb_a/Action_recognition_data/')
parser.add_argument('--samples_extraction', type=bool, default=True)
parser.add_argument('--source_type', type=str, default='opti_flow')
parser.add_argument('--split_type', type=str, default='cross_view')
parser.add_argument('--list_dir', type=str, default='./data_list/')
parser.add_argument('--dataset', type=str, default='NTU')
parser.add_argument('--train_subject', type=int, default=[1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38])
parser.add_argument('--test_subject', type=int, default=[3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40])
parser.add_argument('--train_view', type=int, default=[2, 3])
parser.add_argument('--test_view', type=int, default=[1])
parser.add_argument('--set_size', type=int, default=[256, 310])
parser.add_argument('--frame_train', type=int, default=32)
args = parser.parse_args()

args.list_dir = args.list_dir + '{}/'.format(args.dataset)
samples_save_dir = args.data_root + '{}/{}_jpg'.format(args.dataset, args.source_type)

source_dir = os.path.join(args.data_root, args.dataset, args.source_type)

# if args.samples_extraction:
    # samples_extraction(source_dir, samples_save_dir, args)

create_samples_list(samples_save_dir, args)

