import argparse
from N_UCLA_preprocess import samples_extraction, create_samples_list

parser = argparse.ArgumentParser(description='RGB+D-N_UCLA samples preprocess')
parser.add_argument('--data_root', type=str, default='/home/hkzhang/Documents/sdb_a/Action_recognition_data/')
parser.add_argument('--source_type', type=str, default='opti_flow')
parser.add_argument('--split_type', type=str, default='v3')
parser.add_argument('--list_dir', type=str, default='./data_list/')
parser.add_argument('--dataset', type=str, default='N_UCLA')
parser.add_argument('--set_size', type=int, default=[256, 256])
parser.add_argument('--frame_train', type=int, default=32)
args = parser.parse_args()

if args.split_type == 'v1':
    args.train_view = [2, 3]
    args.test_view = [1]
elif args.split_type == 'v2':
    args.train_view = [1, 3]
    args.test_view = [2]
elif args.split_type == 'v3':
    args.train_view = [1, 2]
    args.test_view = [3]

args.list_dir = args.list_dir + '{}/{}/'.format(args.dataset, args.split_type)
# samples_extraction(args)

samples_save_dir = args.data_root + '{}/{}_jpg'.format(args.dataset, args.source_type)
create_samples_list(samples_save_dir, args)

