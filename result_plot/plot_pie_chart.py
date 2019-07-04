import matplotlib.pyplot as plt

I3D_F = [23.784, 0.371, 29.954, 0.072, 27.586, 16.630, 1.677]
IST_F = [40.168, 0.868, 15.589, 0.072, 26.079, 15.555, 1.736]
SST_F = [44.692, 0.966, 17.345, 0.072, 23.666, 12.112, 1.224]
GSST_F = [61.773, 0.668, 11.990, 0.072, 16.352, 8.367, 0.849]

I3D_P = [0.066, 1.222, 0.004, 5.894, 0.332, 4.754]
IST_P = [0.038, 0.493, 0.004, 2.354, 0.074, 2.103]
SST_P = [0.038, 0.402, 0.004, 1.647, 0.074, 1.333]
GSST_P = [0.038, 0.201, 0.002, 0.824, 0.037, 0.666]

target = 'GSST_F'

labels_F=['C1','C2','C3', 'Max-P', 'MG3', 'MG4', 'MG5']

labels_P=['C1','MG3', 'C2', 'MG4', 'C3', 'MG5']

colors_F=['lightgreen', 'gold', 'lightskyblue', 'lightsteelblue','lightcoral', 'yellowgreen', 'wheat']
colors_P=['lightgreen', 'lightcoral', 'gold', 'yellowgreen', 'lightskyblue','wheat']


if target == 'I3D_F':sizes=I3D_F
elif target == 'IST_F': sizes=IST_F
elif target == 'SST_F': sizes=SST_F
elif target == 'GSST_F': sizes=GSST_F
elif target == 'I3D_P': sizes=I3D_P
elif target == 'IST_P': sizes=IST_P
elif target == 'SST_P': sizes=SST_P
else: sizes=GSST_P

if target.split('_')[-1]=='P':
    labels = labels_P
    colors = colors_P
elif target.split('_')[-1] == 'F':
    labels = labels_F
    colors = colors_F

explode=[0.1,0,0,0,0,0,0]

font = {'family': 'normal',
        'size': 16}
plt.rc('font', **font)

plt.pie(sizes,explode=explode,labels=labels,

        colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)

plt.axis('equal')

im_name = './img/' + '{}.jpg'.format(target)

plt.savefig(im_name)
plt.close()