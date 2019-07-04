import os
import torch
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data.dataloader

from glob import glob
from torch.autograd import Variable


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)


def model_restore(model, trained_model_dir):
    model_list = glob(trained_model_dir + "/*.pkl")
    a = []
    for i in range(len(model_list)):
        index = int(model_list[i].split('model')[-1].split('.')[0])
        a.append(index)
    epoch = np.sort(a)[-1]
    model_path = trained_model_dir + 'trained_model{}.pkl'.format(epoch)
    model.load_state_dict(torch.load(model_path))
    return model, epoch


def model_save(epoch, model, trained_model_dir, max_num = None):
    model_save_dir = trained_model_dir + '/trained_model{}.pkl'.format(epoch)
    model_save = copy.deepcopy(model)
    torch.save(model_save.cpu().state_dict(), model_save_dir)
    if max_num is not None:
        pkl_list = glob(trained_model_dir + '/trained_model*.pkl')
        save_list = np.arange(max(1, epoch-max_num+1), epoch+1)
        for pkl in pkl_list:
            if int(pkl.split('trained_model')[-1].split('.')[0]) not in save_list:
                os.remove(pkl)


def train(epoch, model, device, train_loader, optimizer, criterion, args, lr_scheduler=None):
    if lr_scheduler is not None:
        optimizer = lr_scheduler(epoch, optimizer)
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        train_acc += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()
    length = train_loader.dataset.length
    return (train_acc / length)*100, float(train_loss / length)


def calculate_video_results(output_buffer, video_label, video_id, test_results):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=5)
    top_5 = locs.numpy()
    predict_label = int(top_5[0])
    result_info = {
        'sample': video_id,
        'gt_label': video_label,
        'predict_label': predict_label,
        'top_5': top_5
    }
    test_results.append(result_info)


def val_acc_loss(model, device, val_loader, criterion, args):
    model.eval()
    with torch.no_grad():
        output_buffer = []
        previous_video_id = ''
        previous_video_label = None
        test_results = []
        i = 0
        val_loss = 0
        for data, target, video_ids in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += float(criterion(output, target))
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
        val_loss /= len(val_loader.dataset)

        acc=0
        for result in test_results:
            if result['gt_label'] == result['predict_label']:
                acc += 1
        val_acc = (acc / len(test_results)) * 100

    return val_acc, val_loss

