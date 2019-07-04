import random
import numpy as np


class TemporalECOCrop(object):
    """Split each video into N (frame_set_for_train) segments and randomly select one frame from each segment"""

    def __init__(self, frame_set_for_train):
        self.frame_set = frame_set_for_train

    def __call__(self, frame_num):
        if frame_num < self.frame_set:
            index = np.arange(0, self.frame_set, 1) % frame_num
        elif frame_num > self.frame_set and frame_num < 2* self.frame_set:
            d_start = random.randint(0, frame_num - self.frame_set)
            index = np.arange(d_start, d_start + self.frame_set)
        elif frame_num >= 2 * self.frame_set:
            group_size = int(frame_num//self.frame_set + 1)
            loop_indices = np.arange(0, group_size * self.frame_set) % frame_num
            index_in_loop_indices = np.arange(0, group_size*self.frame_set, step=group_size) \
                                    + np.random.randint(0, group_size, self.frame_set)
            index = loop_indices[index_in_loop_indices]
        return index + 1


class TemporalRandomCrop(object):
    """Densely crop frames of size 'frame_set_for_train' at random location"""

    def __init__(self, frame_set_for_train):
        self.frame_set = frame_set_for_train

    def __call__(self, frame_num):
        if frame_num > self.frame_set:
            d_start = random.randint(0, frame_num - self.frame_set)
            index = np.arange(d_start, d_start + self.frame_set)
        else:
            index = np.arange(0, self.frame_set) % frame_num

        return index + 1
