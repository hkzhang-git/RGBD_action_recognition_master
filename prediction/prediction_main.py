import argparse
import sys
sys.path.append('../training')
sys.path.append('../training/models')
from models_dict import dict
from prediction_utils.data_loader import get_data_loader
from functions_for_prediction import *
from torch.nn import DataParallel

# ***********************************************************************************inference setting
parser = argparse.ArgumentParser(description='models on depth data')
parser.add_argument('--data_list_root', type=str, default='../preprocess/data_list/')
parser.add_argument('--data_root', type=str, default='/home/hkzhang/Documents/sdb_a/Action_recognition_data/')
parser.add_argument('--data_set', type=str, default='NTU')
parser.add_argument('--split_type', type=str, default='cv')
parser.add_argument('--spatial_resolution', type=int, default=224)
parser.add_argument('--model', type=str, default='I3D')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--test_batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--result_save_dir', type=str, default='./prediction_result')
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

# ************************************************************************************data_loaders
rgb_dict_dir = args.data_list_root + '{}/{}/rgb_test.json'.format(args.data_set, args.split_type)
depth_dict_dir = args.data_list_root + '{}/{}/depth_test.json'.format(args.data_set, args.split_type)
opti_flow_dict_dir = args.data_list_root + '{}/{}/opti_flow_test.json'.format(args.data_set, args.split_type)


rgb_loader = get_data_loader(args.data_root,
                             rgb_dict_dir,
                             'rgb',
                             args.spatial_resolution,
                             args.test_batch_size,
                             args.num_workers)


depth_loader = get_data_loader(args.data_root,
                               depth_dict_dir,
                               'depth',
                               args.spatial_resolution,
                               args.test_batch_size,
                               args.num_workers)

opti_flow_loader = get_data_loader(args.data_root,
                                   opti_flow_dict_dir,
                                   'opti_flow',
                                   args.spatial_resolution,
                                   args.test_batch_size,
                                   args.num_workers)

result_save_file = args.result_save_dir + '/{}_{}_{}.txt'.format(args.data_set, args.model, args.split_type)
make_if_not_exist(args.result_save_dir)
device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'

if args.data_set == 'NTU':
    num_cla = 60
elif args.data_set == 'N_UCLA':
    num_cla = 10


rgb_pretrained_model_dir = './trained_models/{}/rgb/{}_{}.pkl'.format(args.data_set, args.model, args.split_type)
depth_pretrained_model_dir = './trained_models/{}/depth/{}_{}.pkl'.format(args.data_set, args.model, args.split_type)
opti_flow_pretrained_model_dir = './trained_models/{}/opti_flow/{}_{}.pkl'.format(args.data_set, args.model, args.split_type)
# *************************************************************************************initial model and optimizer

rgb_model = DataParallel(dict[args.model](num_classes=num_cla, input_channel=3, dropout_keep_prob=0.5))
depth_model = DataParallel(dict[args.model](num_classes=num_cla, input_channel=1, dropout_keep_prob=0.5))
opti_flow_model = DataParallel(dict[args.model](num_classes=num_cla, input_channel=2, dropout_keep_prob=0.5))

label_dict = get_label_dict(rgb_loader)

if args.use_cuda: rgb_model.cuda()
rgb_model.load_state_dict(torch.load(rgb_pretrained_model_dir))
rgb_result = inference(rgb_model, device, rgb_loader)
rgb_model.cpu()

if args.use_cuda:depth_model.cuda()
depth_model.load_state_dict(torch.load(depth_pretrained_model_dir))
depth_result = inference(depth_model, device, depth_loader)
depth_model.cpu()

if args.use_cuda: opti_flow_model.cuda()
opti_flow_model.load_state_dict(torch.load(opti_flow_pretrained_model_dir))
opti_flow_result = inference(opti_flow_model, device, opti_flow_loader)
opti_flow_model.cpu()

result_combine(result_save_file, label_dict, rgb_result, depth_result, opti_flow_result)

