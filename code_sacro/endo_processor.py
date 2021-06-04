#! /usr/bin/env python
import sys
from model import Endo3D
from seq2seq_LSTM import seq2seq_LSTM
from transformer.transformer import Transformer
from utils.sequence_loder import sequence_loder
import numpy as np
import visdom
from tqdm import tqdm

import time
import os
import scipy.io as scio
from scipy import stats
import random

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
from torch.utils import data
import pickle
import json

from sklearn.metrics import confusion_matrix


def phase_f1(seq_true, seq_test):
    seq_true = np.array(seq_true)
    seq_pred = np.array(seq_test)
    index = np.where(seq_true == 0)
    seq_true = np.delete(seq_true, index)
    seq_pred = np.delete(seq_pred, index)
    # f1 = f1_score(seq_true,seq_test,labels=[0, 1, 2, 3, 4, 5], average='weighted')
    # f1 = f1_score(seq_true, seq_test)

    phases = np.unique(seq_true)
    f1s = []
    for phase in phases:
        index_positive_in_true = np.where(seq_true == phase)
        index_positive_in_pred = np.where(seq_pred == phase)
        index_negative_in_true = np.where(seq_true != phase)
        index_negative_in_pred = np.where(seq_pred != phase)

        a = seq_true[index_positive_in_pred]
        unique, counts = np.unique(a, return_counts=True)
        count_dict = dict(zip(unique, counts))
        if phase in count_dict.keys():
            tp = count_dict[phase]
        else:
            tp = 0
        fp = len(index_positive_in_pred[0]) - tp

        b = seq_true[index_negative_in_pred]
        unique, counts = np.unique(b, return_counts=True)
        count_dict = dict(zip(unique, counts))
        if phase in count_dict.keys():
            fn = count_dict[phase]
        else:
            fn = 0
        tn = len(index_negative_in_pred[0]) - fn

        f1 = tp / (tp + 0.5 * (fp + fn))

        f1s.append(f1)

    return sum(f1s) / len(f1s)


def sequence_maker(model, video_name, div, device):
    sacro = sequence_loder(div, device)
    sacro.whole_len_output(video_name)
    whole_loder = data.DataLoader(sacro, 20)
    seq_pre = []
    seq_true = []
    fc_list = []
    for labels_val, inputs_val in whole_loder:
        inputs_val = inputs_val.float().to(device)
        labels_val = labels_val.long().to(device)
        output, x_fc = model.forward_cov(inputs_val)
        _, predicted_labels = torch.max(output.cpu().data, 1)
        for i in range(predicted_labels.numpy().shape[0]):
            seq_pre.append(predicted_labels.numpy()[i])
            seq_true.append(labels_val.cpu().numpy()[i])
            fc_list.append(x_fc[i, :])
    return seq_pre, seq_true, fc_list



path = '/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_sequence/whole'

folders = ['folder1', 'folder2', 'folder3', 'folder4', 'folder5', 'folder6', 'folder7']
div = ['div1', 'div2', 'div3', 'div4', 'div5', 'div6', 'div7']

for counter, folder in enumerate(folders):
    print('The current folder is:' + folder)
    folder_path = os.path.join(path, folder)

    current_path = os.path.abspath(os.getcwd())
    videos_path = os.path.join(current_path, 'data/sacro_jpg')
    with open(os.path.join(videos_path, 'dataset_' + div[counter] + '.json'), 'r') as json_data:
        temp = json.load(json_data)
    video_list = temp['train'] + temp['validation']
    # video_list = temp['test']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Endo3D().to(device)
    Endo3D_state_dict = model.state_dict()
    # pre_state_dict = torch.load('./params/cross_validation/' + div[counter] + '/params_endo3d.pkl')
    pre_state_dict = torch.load('./params/cross_validation/' + div[counter] + '/params_c3d.pkl')
    new_state_dict = {k: v for k, v in pre_state_dict.items() if k in Endo3D_state_dict}
    Endo3D_state_dict.update(new_state_dict)
    model.load_state_dict(Endo3D_state_dict)
    # model.load_state_dict(torch.load('./params/cross_validation/' + div[counter] + '/params_endo3d.pkl'))

    f1 = 0
    for video_name in video_list:
        save_path = os.path.join(folder_path, video_name)

        # os.mkdir(save_path)
        with torch.no_grad():
            seq_pre, seq_true, fc_list = sequence_maker(model, video_name, div[counter], device)

        f1 += phase_f1(seq_true, seq_pre)

        a = open(os.path.join(save_path, 'seq_pred_c3d.pickle'), 'wb')
        pickle.dump(seq_pre, a)
        a.close()
        #
        # b = open(os.path.join(save_path, 'seq_true.pickle'), 'wb')
        # pickle.dump(seq_true, b)
        # b.close()
        #
        # c = open(os.path.join(save_path, 'fc_list.pickle'), 'wb')
        # pickle.dump(fc_list, c)
        # c.close()
    print(div[counter], f1 / len(video_list))






