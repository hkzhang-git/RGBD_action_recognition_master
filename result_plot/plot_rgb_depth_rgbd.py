import matplotlib.pyplot as plt
import numpy as np

root = './txt/'

# I3D
depth_dir = root + 'depth/train_info_cs_I3D_depth.txt'
train_val_info = open(depth_dir).readlines()

d_I3D_epoch = [int(f.split('epoch:')[1].split(',')[0]) for f in train_val_info]
d_I3D_train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in train_val_info]
d_I3D_val_acc = [float(f.split('val_top1:')[1].split(',')[0][:-1]) for f in train_val_info]

#
rgb_dir = root + 'rgb/train_info_cs_I3D_rgb.txt'
train_val_info = open(rgb_dir).readlines()

r_I3D_epoch = [int(f.split('epoch:')[1].split(',')[0]) for f in train_val_info]
r_I3D_train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in train_val_info]
r_I3D_val_acc = [float(f.split('val_top1:')[1].split(',')[0][:-1]) for f in train_val_info]

#
rgbd_dir = root + 'rgbd/train_info_cs_I3D_rgbd.txt'
train_val_info = open(rgbd_dir).readlines()

rd_I3D_epoch = [int(f.split('epoch:')[1].split(',')[0]) for f in train_val_info]
rd_I3D_train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in train_val_info]
rd_I3D_val_acc = [float(f.split('val_top1:')[1].split(',')[0][:-1]) for f in train_val_info]


plt.figure('I3D')
plt.plot(d_I3D_epoch, d_I3D_train_acc, label = 'depth_train_acc', color='dodgerblue', linestyle='-', marker='.')
plt.plot(r_I3D_epoch, r_I3D_train_acc, label = 'rgb_train_acc', color='black', linestyle='-', marker='.')
plt.plot(rd_I3D_epoch, rd_I3D_train_acc, label = 'rgbd_train_acc', color='limegreen', linestyle='-', marker='.')

plt.plot(d_I3D_epoch, d_I3D_val_acc, label = 'depth_val_acc', color='dodgerblue', linestyle='-.', marker='*')
plt.plot(r_I3D_epoch, r_I3D_val_acc, label = 'rgb_val_acc', color='black', linestyle='-.', marker='*')
plt.plot(rd_I3D_epoch, rd_I3D_val_acc, label = 'rgbd_val_acc', color='limegreen', linestyle='-.', marker='*')


plt.grid(True)
plt.legend(loc=4)
plt.axis('tight')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.yticks(np.arange(40, 105, 5))
plt.xticks(np.arange(0, 110, 10))

im_name = './img/' + 'I3D_cs_depth_rgb_rgbd_acc.jpg'

plt.savefig(im_name)
plt.close()



