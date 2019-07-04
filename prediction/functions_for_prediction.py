import os
import torch
import numpy as np
import torch.nn.functional as F


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)


# def calculate_video_results(rgb_output_buffer, depth_output_buffer, opti_flow_output_buffer, video_label, video_id, test_results):
#     rgb_average_scores = torch.mean(torch.stack(rgb_output_buffer, dim=0))
#     depth_average_scores = torch.mean(torch.stack(depth_output_buffer, dim=0))
#     opti_flow_average_scores = torch.mean(torch.stack(opti_flow_output_buffer, dim=0))
#
#     prediction_info = {
#         'video_id': video_id,
#         'video_label': video_label,
#         'rgb_scores': rgb_average_scores,
#         'depth_scores': depth_average_scores,
#         'opti_flow_scores': opti_flow_average_scores
#     }
#
#     test_results.append(prediction_info)


def calculate_video_results(output_buffer, video_label, video_id, test_results):
    average_scores = torch.mean(torch.stack(output_buffer), dim=0)
    test_results.update({video_id: average_scores})


def check_label(gt_label, score):
    sorted_score, locs = torch.topk(score, k=1)
    if int(locs) == gt_label:
        return True
    else:
        return False


def get_label_dict(data_loader):
    previous_video_id = ''
    previous_video_label = None
    label_dict = []
    i = 0
    for data, target, video_ids in data_loader:
        for j in range(len(video_ids)):
            if not (i == 0 and j == 0) and video_ids[j] != previous_video_id:
                info={
                    'video_id': previous_video_id,
                    'video_label': previous_video_label
                }
                label_dict.append(info)

            previous_video_label = int(target[j])
            previous_video_id = video_ids[j]

        i += 1

    info = {
        'video_id': previous_video_id,
        'video_label': previous_video_label
    }
    label_dict.append(info)

    return label_dict


def inference(model, device, data_loader):
    model.eval()

    with torch.no_grad():
        output_buffer = []
        previous_video_id = ''
        previous_video_label = None
        test_results = {}
        i = 0
        for data, target, video_ids in data_loader:
            data = data.to(device)

            output = model(data)
            output_score = F.softmax(output, 1)

            for j in range(len(video_ids)):
                if not (i == 0 and j == 0) and video_ids[j] != previous_video_id:
                    calculate_video_results(output_buffer, previous_video_label, previous_video_id, test_results)
                    output_buffer = []

                output_buffer.append(output_score[j].data.cpu())
                previous_video_label = int(target[j])
                previous_video_id = video_ids[j]

            i += 1
            calculate_video_results(output_buffer, previous_video_label, previous_video_id, test_results)
    return test_results


def val(rgb_model, depth_model, opti_flow_model, device, val_loader):
    rgb_model.eval()
    depth_model.eval()
    opti_flow_model.eval()

    with torch.no_grad():
        rgb_output_buffer = []
        depth_output_buffer = []
        opti_flow_output_buffer = []
        previous_video_id = ''
        previous_video_label = None
        test_results = []
        i = 0
        val_loss = 0
        for rgb_data, depth_data, opti_flow_data, target, video_ids in val_loader:
            rgb_data, depth_data, opti_flow_data, target = rgb_data.to(device), depth_data.to(device), \
                                                           opti_flow_data.to(device), target.to(device)
            rgb_output = rgb_model(rgb_data)
            depth_output = depth_model(depth_data)
            opti_flow_output = opti_flow_model(opti_flow_data)

            rgb_output_score = F.softmax(rgb_output, 1)
            depth_output_score = F.softmax(depth_output, 1)
            opti_flow_output_score = F.softmax(opti_flow_output, 1)

            for j in range(len(video_ids)):
                if not (i == 0 and j == 0) and video_ids[j] != previous_video_id:
                    calculate_video_results(rgb_output_buffer, depth_output_buffer, opti_flow_output_buffer,
                                            previous_video_label, previous_video_id, test_results)

                    rgb_output_buffer = []
                    depth_output_buffer = []
                    opti_flow_output_buffer = []

                rgb_output_buffer.append(rgb_output_score[j].data.cpu())
                depth_output_buffer.append(depth_output_score[j].data.cpu())
                opti_flow_output_buffer.append(opti_flow_output_score[j].data.cpu())

                previous_video_label = int(target[j])
                previous_video_id = video_ids[j]

            i += 1
        calculate_video_results(rgb_output_buffer, depth_output_buffer, opti_flow_output_buffer,
                                previous_video_label, previous_video_id, test_results)
        val_loss /= len(val_loader.dataset)

        rgb_acc = 0
        depth_acc = 0
        opti_flow_acc = 0
        rgb_depth_acc = 0
        rgb_opti_flow_acc = 0
        depth_opti_flow_acc = 0
        rgb_depth_optiflow_acc = 0
        for result in test_results:
            gt_label = result['video_label']

            rgb_score = result['rgb_scores']
            depth_score = result['depth_scores']
            opti_flow_score = result['opti_flow_scores']
            rgb_depth_score = rgb_score + depth_score
            rgb_opti_flow_score = rgb_score + opti_flow_score
            depth_opti_flow_score = depth_score + opti_flow_score
            rgb_depth_optiflow_score = rgb_score + depth_score + opti_flow_score

            if check_label(gt_label, rgb_score): rgb_acc += 1
            if check_label(gt_label, depth_score): depth_acc += 1
            if check_label(gt_label, opti_flow_score): opti_flow_acc += 1
            if check_label(gt_label, rgb_depth_score): rgb_depth_acc += 1
            if check_label(gt_label, rgb_opti_flow_score): rgb_opti_flow_acc += 1
            if check_label(gt_label, depth_opti_flow_score): depth_opti_flow_acc += 1
            if check_label(gt_label, rgb_depth_optiflow_score): rgb_depth_optiflow_acc += 1

        samples_num = len(test_results)

    return (rgb_acc / samples_num)*100,\
           (depth_acc / samples_num)*100, \
           (opti_flow_acc / samples_num)*100, \
           (rgb_depth_acc / samples_num)*100,\
           (rgb_opti_flow_acc / samples_num)*100, \
           (depth_opti_flow_acc / samples_num)*100, \
           (rgb_depth_optiflow_acc / samples_num)*100


def result_combine(result_save_file, label_dict, rgb_result, depth_result, opti_flow_result):
    rgb_acc=0
    depth_acc=0
    opti_flow_acc=0
    rgb_depth_acc=0
    rgb_opti_flow_acc=0
    depth_opti_flow_acc=0
    rgb_depth_opti_flow_acc=0
    count=0

    for item in label_dict:
        video_id = item['video_id']
        video_label = item['video_label']

        if video_id in rgb_result.keys() and video_id in depth_result.keys() and video_id in opti_flow_result.keys():
            if check_label(video_label, rgb_result[video_id]): rgb_acc += 1
            if check_label(video_label, depth_result[video_id]): depth_acc += 1
            if check_label(video_label, opti_flow_result[video_id]): opti_flow_acc += 1
            if check_label(video_label, rgb_result[video_id]+depth_result[video_id]): rgb_depth_acc += 1
            if check_label(video_label, rgb_result[video_id]+opti_flow_result[video_id]): rgb_opti_flow_acc += 1
            if check_label(video_label, depth_result[video_id]+opti_flow_result[video_id]): depth_opti_flow_acc += 1
            if check_label(video_label, rgb_result[video_id]+depth_result[video_id]+opti_flow_result[video_id]): rgb_depth_opti_flow_acc += 1

            count += 1

    with open(result_save_file, 'w') as f:
        f.write('rgb_acc: {}\n'.format(round(rgb_acc/count*100, 2)))
        f.write('depth_acc: {}\n'.format(round(depth_acc/count*100, 2)))
        f.write('opti_flow_acc: {}\n'.format(round(opti_flow_acc/count*100, 2)))
        f.write('rgb_depth_acc: {}\n'.format(round(rgb_depth_acc/count*100, 2)))
        f.write('rgb_opti_flow_acc: {}\n'.format(round(rgb_opti_flow_acc/count*100, 2)))
        f.write('depth_opti_flow_acc: {}\n'.format(round(depth_opti_flow_acc/count*100, 2)))
        f.write('rgb_depth_opti_flow_acc: {}\n'.format(round(rgb_depth_opti_flow_acc/count*100, 2)))






