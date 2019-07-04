import matplotlib.pyplot as plt
import numpy as np

txt_cs = './txt/I3D_rgb_cs.txt'
txt_cv = './txt/I3D_rgb_cv.txt'
txt_cs_ft = './txt/I3D_rgb_ft_cs.txt'
txt_cv_ft = './txt/I3D_rgb_ft_cv.txt'


cs_info = open(txt_cs).readlines()
cv_info = open(txt_cv).readlines()
cs_ft_info = open(txt_cs_ft).readlines()
cv_ft_info = open(txt_cv_ft).readlines()

epoch=[int(f.split('epoch:')[1].split(',')[0]) for f in cs_info]
cs_train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in cs_info]
cs_val_acc = [float(f.split('val_top1:')[1].split(',')[0][:-1]) for f in cs_info]
cv_train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in cv_info]
cv_val_acc = [float(f.split('val_top1:')[1].split(',')[0][:-1]) for f in cv_info]
cs_ft_train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in cs_ft_info]
cs_ft_val_acc = [float(f.split('val_top1:')[1].split(',')[0][:-1]) for f in cs_ft_info]
cv_ft_train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in cv_ft_info]
cv_ft_val_acc = [float(f.split('val_top1:')[1].split(',')[0][:-1]) for f in cv_ft_info]

# cross-subject
plt.figure('cs')

font = {'family': 'normal',
        'size': 12}
plt.rc('font', **font)

plt.plot(epoch, cs_train_acc, label = 'training acc (I3D)', color='dodgerblue', linestyle='-', marker='.')
plt.plot(epoch, cs_ft_train_acc, label = 'training acc (I3D, Kinetics-600 initializing)', color='limegreen', linestyle='-', marker='.')

plt.plot(epoch, cs_val_acc, label = 'test acc (I3D)', color='dodgerblue', linestyle='-.', marker='*')
plt.plot(epoch, cs_ft_val_acc, label = 'test acc (I3D, Kinetics-600 initializing)', color='limegreen', linestyle='-.', marker='*')

plt.grid(True)
plt.legend(loc=4)
plt.axis('tight')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.yticks(np.arange(20, 105, 10))
plt.xticks(np.arange(0, 130, 10))

im_name = './img/' + 'I3D_cs_acc.jpg'

plt.savefig(im_name)
plt.close()

# cross-view
plt.figure('cv')

font = {'family': 'normal',
        'size': 12}
plt.rc('font', **font)

plt.plot(epoch, cv_train_acc, label = 'training acc (I3D)', color='dodgerblue', linestyle='-', marker='.')
plt.plot(epoch, cv_ft_train_acc, label = 'training acc (I3D, Kinetics-600 initializing)', color='limegreen', linestyle='-', marker='.')

plt.plot(epoch, cv_val_acc, label = 'test acc (I3D)', color='dodgerblue', linestyle='-.', marker='*')
plt.plot(epoch, cv_ft_val_acc, label = 'test acc (I3D, Kinetics-600 initializing)', color='limegreen', linestyle='-.', marker='*')

plt.grid(True)
plt.legend(loc=4)
plt.axis('tight')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.yticks(np.arange(40, 105, 10))
plt.xticks(np.arange(0, 130, 10))

im_name = './img/' + 'I3D_cv_acc.jpg'

plt.savefig(im_name)
plt.close()


