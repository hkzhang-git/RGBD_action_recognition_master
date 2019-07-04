import os
import json
import math
import torch
import time
import numpy as np
import torch.utils.data as data
from PIL import Image
from utils.temporal_transforms import TemporalECOCrop, TemporalRandomCrop
from utils.spatial_transforms import (Compose, ToTensor, Rescale, CenterCornerCrop,
                                RandomHorizontalFlip, SpatialRandomCrop)



def load_annotation_data(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


# def pil_loader(path):
#     with open(path, 'rb') as f:
#         with Image.open(f) as img:
#             # return img.convert('RGB')
#             return img


def pil_loader(path):
    return Image.open(path)


def video_loader(video_dir_path, frame_indices, data_type):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{}_{:05d}.jpg'.format(data_type, i))
        if os.path.exists(image_path):
            video.append(pil_loader(image_path))
        else:
            return video

    return video


def opti_flow_video_loader(video_dir_path, frame_indices, data_type):
    video_x = []
    video_y = []
    for i in frame_indices:
        image_x_path = os.path.join(video_dir_path, '{}_x_{:05d}.jpg'.format(data_type, i))
        image_y_path = os.path.join(video_dir_path, '{}_y_{:05d}.jpg'.format(data_type, i))
        if os.path.exists(image_x_path) and os.path.exists(image_y_path):
            video_x.append(pil_loader(image_x_path))
            video_y.append(pil_loader(image_y_path))

    return video_x, video_y


class data_loader(data.Dataset):
    def __init__(self,
                 data_root,
                 dict_dir,
                 temporal_transform=None,
                 spatial_transform=None,
                 model='train',
                 data_type=None):
        self.data_dict = load_annotation_data(dict_dir)
        self.length = len(self.data_dict)
        self.model = model
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.data_root = data_root
        self.data_type = data_type

    def __getitem__(self, index):
        sample_info = self.data_dict[index]
        video_path = os.path.join(self.data_root, sample_info['sample_dir'])
        label = sample_info['action'] - 1
        video_id = sample_info['sample_id']

        if self.model in ['eval', 'test']:
            frame_indices = sample_info['index']
            if self.data_type == 'opti_flow':
                clips_x, clips_y = opti_flow_video_loader(video_path, frame_indices, self.data_type)
                if self.spatial_transform is not None:
                    self.spatial_transform.random_parameters_initialization()
                    clips_x = [self.spatial_transform(img) for img in clips_x]
                    clips_y = [self.spatial_transform(img) for img in clips_y]
                clips_x = torch.stack(clips_x, 0).permute(1, 0, 2, 3)
                clips_y = torch.stack(clips_y, 0).permute(1, 0, 2, 3)
                clips = torch.cat((clips_x, clips_y), 0)
            else:
                clips = video_loader(video_path, frame_indices, self.data_type)
                if self.spatial_transform is not None:
                    self.spatial_transform.random_parameters_initialization()
                    clips = [self.spatial_transform(img) for img in clips]
                clips = torch.stack(clips, 0).permute(1, 0, 2, 3)

        elif self.model=='train':
            frame_indices = self.temporal_transform(sample_info['frame_num'])
            if self.data_type == 'opti_flow':
                clips_x, clips_y = opti_flow_video_loader(video_path, frame_indices, self.data_type)
                if self.spatial_transform is not None:
                    self.spatial_transform.random_parameters_initialization()
                    clips_x = [self.spatial_transform(img) for img in clips_x]
                    clips_y = [self.spatial_transform(img) for img in clips_y]
                clips_x = torch.stack(clips_x, 0).permute(1, 0, 2, 3)
                clips_y = torch.stack(clips_y, 0).permute(1, 0, 2, 3)
                clips = torch.cat((clips_x, clips_y), 0)
            else:
                clips = video_loader(video_path, frame_indices, self.data_type)
                if self.spatial_transform is not None:
                    self.spatial_transform.random_parameters_initialization()
                    self.spatial_transform.random_parameters_initialization()
                    clips = [self.spatial_transform(img) for img in clips]
                clips = torch.stack(clips, 0).permute(1, 0, 2, 3)
        else:
            return

        if self.model == 'train':

            return clips, label
        elif self.model in ['eval', 'test']:
            return clips, label, video_id
        else:
            return

    def __len__(self):
        return self.length


def get_data_loader(data_root, dict_dir, dataset, data_type, spatial_resolution, frame_num, batch_size, test_batch_size, num_workers):

    train_data_dir = dict_dir + 'train.json'
    val_infe_data_dir = dict_dir + 'test.json'

    temporal_transform = TemporalRandomCrop(frame_num)
    train_spatial_transform = Compose([
        RandomHorizontalFlip(),
        CenterCornerCrop(spatial_resolution),
        ToTensor(255)
    ])
    val_spatial_transform = Compose([
        CenterCornerCrop(spatial_resolution, 'c'),
        ToTensor(255)
    ])

    train_loader = torch.utils.data.DataLoader(data_loader(
        data_root=data_root,
        dict_dir=train_data_dir,
        temporal_transform=temporal_transform,
        spatial_transform=train_spatial_transform,
        model='train',
        data_type=data_type),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(data_loader(
        data_root=data_root,
        dict_dir=val_infe_data_dir,
        temporal_transform=None,
        spatial_transform=val_spatial_transform,
        model='test',
        data_type=data_type),
        batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader






