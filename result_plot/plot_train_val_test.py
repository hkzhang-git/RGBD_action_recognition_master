import matplotlib.pyplot as plt

def result_plot(result_txt_dir, flag=0):
    train_val_info = open(result_txt_dir).readlines()
    if len(train_val_info) <= flag:
        return
    epoch = [int(f.split('epoch:')[1].split(',')[0]) for f in train_val_info]
    train_acc = [float(f.split('train_top1:')[1].split(',')[0]) for f in train_val_info]
    train_loss = [float(f.split('train_loss:')[1].split(',')[0]) for f in train_val_info]
    val_acc = [float(f.split('val_top1:')[1].split(',')[0]) for f in train_val_info]
    val_loss = [float(f.split('val_loss:')[1].split(',')[0]) for f in train_val_info]
    test_acc = [float(f.split('test_top1:')[1].split(',')[0]) for f in train_val_info]
    test_loss = [float(f.split('test_loss:')[1].split(',')[0][:-1]) for f in train_val_info]

    plt.figure('acc')
    plt.plot(epoch[flag:], train_acc[flag:], label='train_acc', color='dodgerblue', linestyle='-', marker='.')
    plt.plot(epoch[flag:], val_acc[flag:], label='val_acc', color='black', linestyle='-', marker='.')
    plt.plot(epoch[flag:], test_acc[flag:], label='test_acc', color='limegreen', linestyle='-', marker='.')
    plt.grid(True)
    plt.legend(loc=4)
    plt.axis('tight')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    im_name = './img/' + result_txt_dir.split('/')[-1].split('.')[0] + '_acc.jpg'
    plt.savefig(im_name)
    plt.close()

    plt.figure('loss')
    plt.plot(epoch[flag:], train_loss[flag:], label='train_loss', color='dodgerblue', linestyle='-', marker='.')
    plt.plot(epoch[flag:], val_loss[flag:], label='val_loss', color='black', linestyle='-', marker='.')
    plt.plot(epoch[flag:], test_loss[flag:], label='test_loss', color='limegreen', linestyle='-', marker='.')
    plt.grid(True)
    plt.legend(loc=1)
    plt.axis('tight')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    im_name = './img/' + result_txt_dir.split('/')[-1].split('.')[0] + '_loss.jpg'
    plt.savefig(im_name)
    plt.close()

if __name__=='__main__':
    result_plot('./txt/train_info_S3D_P_rgb_adam.txt')
