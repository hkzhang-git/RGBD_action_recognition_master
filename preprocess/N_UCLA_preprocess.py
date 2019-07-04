import os
import json
import time
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
    im=im/3975.0
    im[np.where(im>1.0)]=1.0
    return np.uint8(im*255)


def action_2_label(action_num):
    if action_num <= 7:
        return action_num
    elif action_num <= 10:
        return action_num-1
    elif action_num <= 12:
        return action_num-2


def png_to_jpg(depth_list, rgb_list, frame_num, depth_save_dir, rgb_save_dir, set_size, sample_name, v_id):

    if frame_num ==0:
        return

    sample_name = v_id + '_' + sample_name
    depth_sample_save_dir = os.path.join(depth_save_dir, sample_name)
    rgb_sample_save_dir = os.path.join(rgb_save_dir, sample_name)
    make_if_not_exist(depth_sample_save_dir)
    make_if_not_exist(rgb_sample_save_dir)

    frame_num = frame_num
    for frame_id in range(frame_num):
        depth_dir = depth_list[frame_id]
        rgb_dir = rgb_list[frame_id]

        depth_im = np.array(Image.open(depth_dir).resize((set_size[1], set_size[0])))
        depth_im_rescaled = Image.fromarray(img_value_rescale(depth_im))
        depth_im_rescaled.save(depth_sample_save_dir + '/depth_{:05d}.jpg'.format(frame_id))

        rgb_im = Image.open(rgb_dir).resize((set_size[1], set_size[0]))
        rgb_im.save(rgb_sample_save_dir + '/rgb_{:05d}.jpg'.format(frame_id))


def subset_to_jpeg(id, subset_dir, filelist_in_subset,
                   depth_save_dir, rgb_save_dir, set_size):

    view_id = subset_dir.split('/')[-1]
    sample_name = filelist_in_subset[id]
    sample_dir = os.path.join(subset_dir, sample_name)
    depth_png_list = glob(sample_dir + '/frame_*_tc_*_depth.png')

    frame_list = [int(f.split('/')[-1].split('_')[1]) for f in depth_png_list]
    frame_list = np.sort(np.array(frame_list))

    depth_list = []
    rgb_list = []
    for frame_id in frame_list:
        depth_png = glob(sample_dir + '/frame_{}_tc_*_depth.png'.format(frame_id))
        rgb_jpg = glob(sample_dir + '/frame_{}_tc_*_rgb.jpg'.format(frame_id))
        if os.path.exists(depth_png[0]): depth_list.append(depth_png[0])
        if os.path.exists(rgb_jpg[0]): rgb_list.append(rgb_jpg[0])

    assert len(depth_list) == len(rgb_list)
    frame_num = len(depth_list)

    png_to_jpg(depth_list, rgb_list, frame_num, depth_save_dir, rgb_save_dir, set_size, sample_name, view_id)


def samples_extraction(args):
    source_dir = args.data_root + '{}/depth_rgb/'.format(args.dataset)
    save_dir = args.data_root + '{}/'.format(args.dataset)
    set_size = args.set_size
    list_dir = args.list_dir + '{}/'.format(args.split_type)
    make_if_not_exist(list_dir)

    depth_save_dir = save_dir + 'depth_jpg'
    rgb_save_dir = save_dir + 'rgb_jpg'
    make_if_not_exist(depth_save_dir)
    make_if_not_exist(rgb_save_dir)

    subset_list=os.listdir(source_dir)

    print('start preprocess: there are {} samples'.format(len(subset_list)))
    count=0
    for subset in subset_list:
        count+=1
        subset_dir = source_dir+subset
        filelist_in_subset = os.listdir(subset_dir)

        start_time = time.time()
        pool = Pool()
        partial_subset_to_jpeg = partial(subset_to_jpeg, subset_dir=subset_dir,
                                         filelist_in_subset=filelist_in_subset, depth_save_dir=depth_save_dir,
                                         rgb_save_dir=rgb_save_dir, set_size=set_size)
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


def action_2_label(action_num):
    if action_num <= 7:
        return action_num
    elif action_num <= 10:
        return action_num-1
    elif action_num <= 12:
        return action_num-2


def create_samples_list(sample_save_dir, args):

    list_dir = args.list_dir
    list_save_dir = list_dir
    make_if_not_exist(list_save_dir)

    train_id_list = np.array(args.train_view)
    test_id_list = np.array(args.test_view)

    sample_list = os.listdir(sample_save_dir)
    sample_list.sort()

    train_dict = []
    test_dict = []

    for sample in sample_list:
        sample_info = sample.split('_')
        subject_id = '_'.join(sample_info[2:])
        view_id = int(sample_info[1])
        action = int(sample_info[2][1:])

        sample_dir = '{}/{}_jpg/{}'.format(args.dataset, args.source_type, sample)
        if args.source_type == 'opti_flow':
            frame_num = glob(args.data_root + '{}/{}_jpg/{}'.format(args.dataset, args.source_type, sample) + '/{}_x_*.jpg'.format(args.source_type)).__len__()
        else:
            frame_num = glob(args.data_root + '{}/{}_jpg/{}'.format(args.dataset, args.source_type, sample) + '/{}_*.jpg'.format(args.source_type)).__len__() - 1

        if view_id in train_id_list:
            sample_info = {
                'sample_id': subject_id,
                'sample_dir': sample_dir,
                'action': action_2_label(action),
                'frame_num': frame_num
            }
            train_dict.append(sample_info)
        elif view_id in test_id_list:
            index = video_split(frame_num, args.frame_train)

            for i in index:
                if len(index) == 1:
                    frame_index_npy = np.array((range(0, args.frame_train))) % frame_num + 1
                    frame_index = [int(item) for item in frame_index_npy]
                else:
                    frame_index = list(range(i+1, i+args.frame_train+1))
                sample_info = {
                    'sample_id': subject_id,
                    'sample_dir': sample_dir,
                    'action': action_2_label(action),
                    'index': frame_index
                }
                test_dict.append(sample_info)

    train_dict_save_dir = list_save_dir + '{}_train.json'.format(args.source_type)
    test_dict_save_dir = list_save_dir + '{}_test.json'.format(args.source_type)
    with open(train_dict_save_dir, 'w') as dst_file:
        json.dump(train_dict, dst_file)
    with open(test_dict_save_dir, 'w') as dst_file:
        json.dump(test_dict, dst_file)

