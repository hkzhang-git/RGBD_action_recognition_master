import time
import torch
import argparse
import sys
sys.path.append('./models')
from models_dict import dict
from data_loader_1 import data_loader
from functions_for_time_test import *
from torch.nn import DataParallel
from torch.autograd import Variable

# ***********************************************************************************Training settings
parser = argparse.ArgumentParser(description='models on rgb data')
parser.add_argument('--data_list_root', type=str, default='../preprocess/data_list/')
parser.add_argument('--data_set', type=str, default='N_UCLA')
parser.add_argument('--model_split', type=str, default='cs')
parser.add_argument('--split_type', type=str, default='v3')
parser.add_argument('--sample_shape', type=int, default=[224, 224, 32])
parser.add_argument('--model_name', type=str, default='ITS_1')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--devices', type=str, default='0')
parser.add_argument('--test_batch_size', type=int, default=4)
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=2)
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

# ************************************************************************************data loaders
rgb_data_dir = args.data_list_root + '{}/{}_{}_rgb_test.txt'.format(args.data_set, args.data_set, args.split_type)
rgb_eval_data_dir = args.data_list_root + '{}/{}_{}_rgb_test_eval.txt'.format(args.data_set, args.data_set, args.split_type)

rgb_val_loader = torch.utils.data.DataLoader(data_loader(rgb_eval_data_dir, args.sample_shape, 'rgb', 'eval')
    , batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

# *************************************************************************************initial model and optimizer
model_rgb = DataParallel(dict[args.model_name](num_classes=10, input_channel=3, dropout_keep_prob=0.0))
# model_rgb = dict[args.model_name](num_classes=10, input_channel=3, dropout_keep_prob=0.0)
if args.use_cuda: model_rgb.cuda()


for data, target, video_ids in rgb_val_loader:
    if args.use_cuda:
        data = data.cuda()
    data = Variable(data / 127.5 - 1, volatile=True)
    time_start=time.time()
    output = model_rgb(data)
    torch.cuda.synchronize()
    time_cost = time.time()-time_start
    print(time_cost)


