import matplotlib.pyplot as plt
import numpy as np


train_val_info_dir = './txt/train_info_I3D.txt'
# test_info_dir = 'test_info.txt'
train_val_info = open(train_val_info_dir).readlines()
# test_info = open(test_info_dir).readlines()

# plot acc**************************************************************************
epoch = [int(f.split('epoch:')[1].split(',')[0]) for f in train_val_info]
train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in train_val_info]
val_acc = [float(f.split('val_top1:')[1].split(',')[0][:-1]) for f in train_val_info]
# train_acc = [float(f.split('train_acc:')[1].split(',')[0]) for f in train_val_info]
# val_acc = [float(f.split('val_acc:')[1].split(',')[0][:-1]) for f in train_val_info]

# epoch_test = [int(f.split('trained_model')[1].split('.')[0]) for f in test_info]
# test_acc = [float(f.split('train_acc:')[1].split(',')[0][:-1]) for f in test_info]


plt.figure(train_val_info_dir[:-4])
plt.plot(epoch, train_acc, label = 'train_acc')
plt.plot(epoch, val_acc, label = 'val_acc')
plt.plot(epoch, train_acc, 'g*')
plt.plot(epoch, val_acc, 'b*')
# plt.plot(epoch_test, test_acc, 'r*')
plt.grid(True)
plt.legend(loc=4)
plt.axis('tight')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.yticks(np.arange(40, 105, 5))
plt.xticks(np.arange(0, 110, 10))

im_name = './img/' + train_val_info_dir[5:-4] + '_acc.jpg'

plt.savefig(im_name)
plt.close()


# # plot loss *************************************************************************
# epoch = [int(f.split('epoch:')[1].split(',')[0]) for f in train_val_info]
# train_acc = [float(f.split('train_loss:')[1].split(',')[0]) for f in train_val_info]
# val_acc = [float(f.split('val_loss:')[1].split(',')[0]) for f in train_val_info]
#
# # epoch_test = [int(f.split('trained_model')[1].split('.')[0]) for f in test_info]
# # test_acc = [float(f.split('train_acc:')[1].split(',')[0][:-1]) for f in test_info]
#
#
# plt.figure(train_val_info_dir[:-4])
# plt.plot(epoch, train_acc, label = 'train_loss')
# plt.plot(epoch, val_acc, label = 'val_loss')
# # plt.plot(epoch_test, test_acc, label = 'test_acc')
# plt.plot(epoch, train_acc, 'g*')
# plt.plot(epoch, val_acc, 'b*')
# # plt.plot(epoch_test, test_acc, 'r*')
# plt.grid(True)
# plt.legend(loc=3)
# plt.axis('tight')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.yticks(np.arange(0.5, 5, 0.5))
# plt.xticks(np.arange(0, 160, 10))
#
# im_name = './img/' + train_val_info_dir[5:-4] + '_loss.jpg'
#
# plt.savefig(im_name)
