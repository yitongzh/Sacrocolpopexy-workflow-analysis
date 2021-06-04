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


def transition_filter(candi_seq, threshold=10):
    chunk_starts = []
    chunk_ends = []
    for i in range(len(candi_seq) - 1):
        if candi_seq[i] != candi_seq[i + 1] and candi_seq[i + 1] == 0:
            chunk_starts.append(i + 1)
        if candi_seq[i] != candi_seq[i + 1] and candi_seq[i] == 0:
            chunk_ends.append(i)
    is_rewrite = [True if chunk_ends[i] - chunk_starts[i] < threshold else False for i in range(len(chunk_starts))]
    for i in range(len(chunk_starts)):
        if is_rewrite[i]:
            for j in range(chunk_starts[i], chunk_ends[i] + 1):
                # candi_seq[j] = candi_seq[chunk_starts[i] - 1]
                candi_seq[j] = candi_seq[chunk_ends[i] + 1]
    return candi_seq


test_video = 1
fol_num = 6
folders = ['div1', 'div2', 'div3', 'div4', 'div5', 'div6', 'div7']
KF_Folders = ['folder1', 'folder2', 'folder3', 'folder4', 'folder5', 'folder6', 'folder7']

vis = visdom.Visdom(env='sequence_tester')
current_path = os.path.abspath(os.getcwd())
videos_path = os.path.join(current_path, 'data/sacro_jpg')

cm_diff = np.zeros((6, 6))
for fol_num in range(1):
    for test_video in range(2):
        # test_video = 1
        # fol_num = 6
        with open(os.path.join(videos_path, 'dataset_' + folders[fol_num] + '.json'), 'r') as json_data:
            temp = json.load(json_data)
        test_video_list = temp['test']
        validation_video_list = temp['validation']

        video_name = test_video_list[test_video]
        open_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_sequence/whole',
                                 KF_Folders[fol_num], video_name)

        file = open(os.path.join(open_path, 'seq_pred.pickle'), 'rb')
        seq_pre = pickle.load(file)
        file.close()

        file = open(os.path.join(open_path, 'seq_true.pickle'), 'rb')
        seq_true = pickle.load(file)
        file.close()

        file = open(os.path.join(open_path, 'seq_pred_c3d.pickle'), 'rb')
        seq_pre_c3d = pickle.load(file)
        file.close()

        cm1 = confusion_matrix(np.array(seq_true), np.array(seq_pre_c3d), labels=[0, 1, 2, 3, 4, 5])
        # seq_true = transition_filter(seq_true)
        cm2 = confusion_matrix(np.array(seq_true), np.array(seq_pre_c3d), labels=[0, 1, 2, 3, 4, 5])
        cm_diff += cm2-cm1
        # vis.heatmap(X=cm1, win='heatmap1', opts=dict(title='confusion matrix sequential_before',
        #                                              rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
        #                                              columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
        # vis.heatmap(X=cm2, win='heatmap2', opts=dict(title='confusion matrix sequential2_after',
        #                                              rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
        #                                              columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
        # vis.heatmap(X=cm2 - cm1, win='heatmap3', opts=dict(title='confusion matrix sequential2_diff',
        #                                                    rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
        #                                                    columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
        #
        vis.close(win='Surgical workflow MODE')
        vis.line(X=np.array(range(len(seq_pre))),
                 Y=np.column_stack((np.array(seq_pre), np.array(seq_pre_c3d), np.array(seq_true))),
                 win='Surgical workflow MODE', update='append',
                 opts=dict(title='Surgical workflow MODE', showlegend=True, legend=['Prediction', 'c3d', 'Ground Truth']))
vis.heatmap(X=cm_diff, win='heatmap3', opts=dict(title='confusion matrix sequential2_diff',
                                                   rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
                                                   columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
