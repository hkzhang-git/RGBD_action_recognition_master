import os
import cv2
import numpy as np


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def truncate(flow, min=-8, max=8):
    flow[flow>max]=max
    flow[flow<min]=min
    t_flow= np.uint8((flow + min) * (255/(max-min)))
    return t_flow


def optical_flow_extraction(save_path, source_path, set_size):
    subset_id = source_path.split('/')[-2]
    video_id = source_path.split('/')[-1].split('_')[0]

    optical_save_path = os.path.join(save_path, subset_id, video_id)
    make_if_not_exist(optical_save_path)

    cap = cv2.VideoCapture(source_path)
    img_h = cap.get(4)
    img_w = cap.get(3)

    if img_h and img_w:
        data_npy = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (set_size[1], set_size[0]))
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                data_npy.append(frame_gray)
            else:
                break

    optical_flow = cv2.createOptFlow_DualTVL1()
    # optical_flow = cv2.DualTVL1OpticalFlow_create()

    flow_npy = []
    for frame_id in range(1, len(data_npy)):
        # flow = cv2.calcOpticalFlowFarneback(data_npy[frame_id-1], data_npy[frame_id], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = optical_flow.calc(data_npy[frame_id-1], data_npy[frame_id], None)
        t_flow = truncate(flow)

        cv2.imwrite(optical_save_path + '/opti_flow_x_' + '%05d' % frame_id + '.jpg', t_flow[..., 0])
        cv2.imwrite(optical_save_path + '/opti_flow_y_' + '%05d' % frame_id + '.jpg', t_flow[..., 1])


def subset_opti_flow(id, save_path, video_list, set_size):
    optical_flow_extraction(save_path, video_list[id], set_size)



