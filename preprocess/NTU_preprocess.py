import os
import cv2
import time
import json
import numpy as np
from glob import glob
from PIL import Image
from functools import partial
from multiprocessing import Pool


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)


def Image_resize(set_size, jpg_path, jpg_save_dir):
    im=Image.open(jpg_path).resize((int(set_size*1.15), int(set_size*1.15)))
    im.save(jpg_save_dir)


def img_value_rescale(im):
    im=im/5000.0
    im[np.where(im>1.0)]=1.0
    return np.uint8(im*255)


def png_to_jpeg(file_path, frame_num, sample_save_dir, set_size):
    frame_num = frame_num
    frame_id_list = ["%08d" % int(num) for num in range(2, frame_num+1)]
    id = 1
    for frame_id in frame_id_list:
        frame_dir = file_path+'/MDepth-{}.png'.format(frame_id)
        im = np.array(Image.open(frame_dir).resize((set_size[1], set_size[0])))
        im_rescaled = Image.fromarray(img_value_rescale(im))
        im_rescaled.save(sample_save_dir + '/depth_{:05d}.jpg'.format(id))
        id += 1


def avi_to_jpeg(avi_path, save_dir, filename, set_size):
    cap = cv2.VideoCapture(avi_path)
    img_h = cap.get(4)
    img_w = cap.get(3)

    if img_h and img_w:
        jpg_save_dir = os.path.join(save_dir, filename)
        make_if_not_exist(jpg_save_dir)
        f_n = 0
        data_npy = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame_i = cv2.resize(frame, (set_size[1], set_size[0]))
                data_npy.append(frame_i)
                f_n += 1
            else:
                break

        for i in range(1, f_n):
            cv2.imwrite(jpg_save_dir + '/rgb_{:05d}.jpg'.format(i), data_npy[i])

    else:
        cap.release()
        return


def subset_to_jpeg(id, subset_dir, filelist_in_subset, save_dir, set_size, source_type):
    if source_type == 'depth':
        filename = filelist_in_subset[id]
        file_path = subset_dir + '/' + filename
        sample_save_dir = os.path.join(save_dir, filename)
        make_if_not_exist(sample_save_dir)
        frame_num = len(glob(file_path + '/MDepth-*.png'))
        png_to_jpeg(file_path, frame_num, sample_save_dir, set_size)
    elif source_type == 'rgb':
        avi_path = filelist_in_subset[id]
        filename = avi_path.split('/')[-1].split('_rgb')[0]
        avi_to_jpeg(avi_path, save_dir, filename, set_size)
    else:
        return


def samples_extraction(source_dir, save_dir, args):

    set_size = args.set_size
    source_type = args.source_type
    make_if_not_exist(save_dir)
    subset_list=os.listdir(source_dir)

    print('start preprocess: there are {} samples'.format(len(subset_list)))
    count=0
    for subset in subset_list:
        count+=1
        subset_dir = os.path.join(source_dir, subset)
        if source_type=='depth':
            filelist_in_subset = os.listdir(subset_dir)
        elif source_type=='rgb':
            filelist_in_subset = glob(subset_dir + '/*.avi')
        else:
            print('undefined type: {}'.format(source_type))
            return

        start_time = time.time()
        pool = Pool()
        partial_subset_to_jpeg = partial(subset_to_jpeg, subset_dir=subset_dir, filelist_in_subset=filelist_in_subset,
                                         save_dir=save_dir, set_size=set_size, source_type=source_type)
        N = len(filelist_in_subset)
        _ = pool.map(partial_subset_to_jpeg, range(N))
        pool.close()
        pool.join()
        end_time = time.time()
        print('subset {:}:{} is done, cost {:.2f} seconds'.format(count, subset, end_time - start_time))


def video_split(frame_num, frame_train):
    if frame_num <=frame_train:
        return [0]
    else:
        index=[]
        [index.append(num) for num in np.arange(0, frame_num-frame_train, frame_train)]
        index.append(frame_num-frame_train-1)
        return index


def create_samples_list(sample_save_dir, args):
    if args.split_type == 'cross_subject':
        split_type = 'cs'
    elif args.split_type == 'cross_view':
        split_type = 'cv'

    list_dir = args.list_dir
    list_save_dir = list_dir + '{}/'.format(split_type)
    make_if_not_exist(list_save_dir)

    if args.split_type == 'cross_subject':
        train_id_list = np.array(args.train_subject)
        test_id_list = np.array(args.test_subject)
    elif args.split_type == 'cross_view':
        train_id_list = np.array(args.train_view)
        test_id_list = np.array(args.test_view)
    else:
        return

    sample_list = os.listdir(sample_save_dir)
    sample_list.sort()

    train_dict = []
    test_dict = []

    for sample in sample_list:
        subject_id = int(sample.split('P')[1][:3])
        view_id = int(sample.split('C')[1][:3])
        action = int(sample[-3:])
        sample_dir = '{}/{}_jpg/{}'.format(args.dataset, args.source_type, sample)
        if args.source_type == 'opti_flow':
            frame_num = glob(args.data_root + '{}/{}_jpg/{}'.format(args.dataset, args.source_type, sample) + '/{}_x_*.jpg'.format(args.source_type)).__len__()
        else:
            frame_num = glob(args.data_root + '{}/{}_jpg/{}'.format(args.dataset, args.source_type, sample) + '/{}_*.jpg'.format(args.source_type)).__len__()


        if args.split_type == 'cross_subject':
            split_switch = subject_id
        elif args.split_type == 'cross_view':
            split_switch = view_id

        if split_switch in train_id_list:
            sample_info = {
                'sample_id': sample,
                'sample_dir': sample_dir,
                'action': action,
                'frame_num': frame_num
            }
            train_dict.append(sample_info)
        elif split_switch in test_id_list:
            index = video_split(frame_num, args.frame_train)

            for i in index:
                if len(index) == 1:
                    frame_index_npy = np.array((range(0, args.frame_train))) % frame_num + 1
                    frame_index = [int(item) for item in frame_index_npy]
                else:
                    frame_index = list(range(i+1, i+args.frame_train+1))
                sample_info = {
                    'sample_id': sample,
                    'sample_dir': sample_dir,
                    'action': action,
                    'index': frame_index
                }
                test_dict.append(sample_info)

    train_dict_save_dir = list_save_dir + '{}_train.json'.format(args.source_type)
    test_dict_save_dir = list_save_dir + '{}_test.json'.format(args.source_type)
    with open(train_dict_save_dir, 'w') as dst_file:
        json.dump(train_dict, dst_file)
    with open(test_dict_save_dir, 'w') as dst_file:
        json.dump(test_dict, dst_file)















