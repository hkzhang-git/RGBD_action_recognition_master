import matplotlib.pyplot as plt
import numpy as np

root = './txt/rgb/'

# I3D
train_val_info_dir = root + 'train_info_I3D_rgb.txt'
train_val_info = open(train_val_info_dir).readlines()

I3D_epoch = [int(f.split('epoch:')[1].split(',')[0]) for f in train_val_info]
I3D_train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in train_val_info]
I3D_val_acc = [float(f.split('val_top1:')[1].split(',')[0][:-1]) for f in train_val_info]

# S3D
train_val_info_dir = root + 'train_info_S3D_rgb.txt'
train_val_info = open(train_val_info_dir).readlines()

S3D_epoch = [int(f.split('epoch:')[1].split(',')[0]) for f in train_val_info]
S3D_train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in train_val_info]
S3D_val_acc = [float(f.split('val_top1:')[1].split(',')[0][:-1]) for f in train_val_info]

# SS3D
train_val_info_dir = root + 'train_info_SS3D_2_rgb.txt'
train_val_info = open(train_val_info_dir).readlines()

SS3D_epoch = [int(f.split('epoch:')[1].split(',')[0]) for f in train_val_info]
SS3D_train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in train_val_info]
SS3D_val_acc = [float(f.split('val_top1:')[1].split(',')[0][:-1]) for f in train_val_info]

# GSS3D
train_val_info_dir = root + 'train_info_GSS3D_2_rgb.txt'
train_val_info = open(train_val_info_dir).readlines()

GSS3D_epoch = [int(f.split('epoch:')[1].split(',')[0]) for f in train_val_info]
GSS3D_train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in train_val_info]
GSS3D_val_acc = [float(f.split('val_top1:')[1].split(',')[0][:-1]) for f in train_val_info]

plt.figure('I3D_S3D_SS3D_GSS3D')
plt.plot(I3D_epoch, I3D_train_acc, label = 'I3D_train_acc', color='dodgerblue', linestyle='-', marker='.')
plt.plot(S3D_epoch, S3D_train_acc, label = 'S3D_train_acc', color='black', linestyle='-', marker='.')
plt.plot(SS3D_epoch, SS3D_train_acc, label = 'SS3D_train_acc', color='limegreen', linestyle='-', marker='.')
plt.plot(GSS3D_epoch, GSS3D_train_acc, label = 'GSS3D_train_acc', color='cyan', linestyle='-', marker='.')

plt.plot(I3D_epoch, I3D_val_acc, label = 'I3D_val_acc', color='dodgerblue', linestyle='-.', marker='*')
plt.plot(S3D_epoch, S3D_val_acc, label = 'S3D_val_acc', color='black', linestyle='-.', marker='*')
plt.plot(SS3D_epoch, SS3D_val_acc, label = 'SS3D_val_acc', color='limegreen', linestyle='-.', marker='*')
plt.plot(GSS3D_epoch, GSS3D_val_acc, label = 'GSS3D_val_acc', color='cyan', linestyle='-.', marker='*')

plt.plot([0, 110], [87.08, 87.08], label = '(depth)TMM 2018 DDMNI', color='red', linestyle='--')
plt.plot([0, 110], [84.8, 84.8], label = '(skeleton)arxiv 2018-6-29 MvdiCNN', color='darkorchid', linestyle='--')
plt.plot([0, 110], [81.5, 81.5], label = '(rgb)AAAI 2018 ST-GCN', color='darkred', linestyle='--')

plt.grid(True)
plt.legend(loc=4)
plt.axis('tight')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.yticks(np.arange(40, 105, 5))
plt.xticks(np.arange(0, 110, 10))

im_name = './img/' + 'I3D_S3D_SS3D2_GSS3D2_rgb_acc.jpg'

plt.savefig(im_name)
plt.close()



