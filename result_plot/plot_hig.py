import matplotlib.pyplot as plt

name_list=['I3D', 'IST', 'SST', 'GSST']

NTU_acc =[94.3, 95.2, 95.2, 94.9]
NUCLA_acc=[95.9, 96.6, 97.2, 95.3]
Param = [12.27, 5.07, 3.50, 1.77]
FLOPs = [55.92, 24.07, 21.41, 15.54]

im_name = './img/' + 'NUCLA_acc.jpg'
data=NUCLA_acc

index=[0, 1, 2, 3]

font = {'family': 'normal',
        'size': 15}
plt.rc('font', **font)

plt.ylim(ymax=100, ymin=90)
plt.xticks(index, name_list)
plt.ylabel('Accuracy')
plt.xlabel('Models')


colors=['cyan', 'yellowgreen', 'dodgerblue', 'lightskyblue']
for i in range(len(colors)):
    plt.bar(left=index[i], height=data[i], facecolor=colors[i], width=0.4, align='center', )
    plt.text(index[i]-0.2, data[i]+0.3, '{}'.format(data[i]))


plt.savefig(im_name)
plt.close()
