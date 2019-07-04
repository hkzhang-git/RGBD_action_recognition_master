import os
import time
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
from tensorboardX import SummaryWriter
from models.models_dict import dict
from functions_for_training import *
from utils.lr_scheduler import MultiStepLR
from utils.data_loader import get_data_loader


# ***********************************************************************************Training settings
parser = argparse.ArgumentParser(description='models on depth data')
parser.add_argument('--data_list_root', type=str, default='../preprocess/data_list/')
parser.add_argument('--data_root', type=str, default='/home/hkzhang/Documents/sdb_a/Action_recognition_data/')
parser.add_argument('--data_set', type=str, default='NTU')
parser.add_argument('--data_type', type=str, default='rgb')
parser.add_argument('--split_type', type=str, default='cv')
parser.add_argument('--spatial_resolution', type=int, default=224)
parser.add_argument('--frame_num', type=int, default=32)
parser.add_argument('--model', type=str, default='IST')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--restore', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--test_batch_size', type=int, default=3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr_scheduler', type=list, default=[50, 85])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.00001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--model_save_interval', type=int, default=1)
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

# ************************************************************************************data_loaders
dict_dir = args.data_list_root + '{}/{}/{}_'.format(args.data_set, args.split_type, args.data_type)


train_loader, val_loader = get_data_loader(args.data_root,
                                           dict_dir,
                                           args.data_set,
                                           args.data_type,
                                           args.spatial_resolution,
                                           args.frame_num,
                                           args.batch_size,
                                           args.test_batch_size,
                                           args.num_workers)

torch.manual_seed(args.seed)
if args.use_cuda and torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'

trained_model_dir = './train_info/' + '{}/{}_{}/{}_{}_{}_lr{}_wd{}_adam/'\
    .format(args.data_set, args.split_type, args.data_type, args.model, args.spatial_resolution, args.frame_num, args.lr, args.weight_decay)

make_if_not_exist(trained_model_dir)

# *****************************************************************************model and optimizer initialization
if args.data_set == 'NTU':
    num_cla = 60
elif args.data_set == 'N-UCLA':
    num_cla = 10
else:
    print('undefined data_set *_*!')

if args.data_type == 'depth':
    input_cha = 1
elif args.data_type == 'opti_flow':
    input_cha = 2
elif args.data_type == 'rgb':
    input_cha = 3

model = DataParallel(dict[args.model](num_classes=num_cla, input_channel=input_cha, dropout_keep_prob=0.5))

if args.use_cuda: model.cuda()
cudnn.benchmark = True

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

criterion = nn.CrossEntropyLoss()
lr_scheduler = MultiStepLR(optimizer, args.lr_scheduler)

# **************************************************************************************resume model
start_epoch = 0
if args.restore and len(os.listdir(trained_model_dir)):
    model, start_epoch = model_restore(model, trained_model_dir)

# **************************************************************************************training
writer = SummaryWriter(log_dir=trained_model_dir + 'log', comment='log_file')

for epoch in range(start_epoch+1, args.epochs+1):
    start = time.time()
    train_acc, train_loss = train(epoch, model, device, train_loader, optimizer, criterion, args, lr_scheduler)
    end = time.time()
    print('train_acc:{:.2f}% train_loss:{:.6f}, lr: {:.6f}, epoch: {}, time_cost: {:.2f} min'
          .format(train_acc, train_loss, optimizer.param_groups[0]['lr'], epoch, (end-start)/60))
    # save model
    model_save(epoch, model, trained_model_dir, 50)

    if epoch % args.model_save_interval == 0:
        val_start_time = time.time()
        val_acc, val_loss = val_acc_loss(model, device, val_loader, criterion, args)
        val_end_time = time.time()

        print('val_acc:{:.2f}% val_loss:{:.6f}, epoch: {}, time_cost: {:.2f} min'
              .format(val_acc, val_loss, epoch, (val_end_time-val_start_time)/60))

        writer.add_scalars('time_cost', {'train_time_cost': (end - start) / 60,
                                         'val_time_cost': (val_end_time - val_start_time) / 60},
                           epoch)
        writer.add_scalars('loss', {'train_loss': train_loss,
                                    'val_loss': val_loss},
                           epoch)
        writer.add_scalars('acc', {'train_loss': train_acc,
                                    'val_acc': val_acc},
                           epoch)
