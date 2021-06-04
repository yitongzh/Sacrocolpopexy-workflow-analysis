#! /usr/bin/env python
import sys
from model import Endo3D
from seq2seq_LSTM import seq2seq_LSTM
from transformer.transformer import Transformer
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


def mode_average(prediction, slinwin_sz=17):
    results = []
    hf = int((slinwin_sz - 1) / 2)
    prediction_pad = np.hstack((np.full([hf, ], np.nan), prediction))
    prediction_pad = np.hstack((prediction_pad, np.full([hf, ], np.nan)))
    for i in range(prediction.shape[0]):
        st_idx = i
        ed_idx = i + slinwin_sz - 1
        results.append(stats.mode(prediction_pad[st_idx:ed_idx])[0][0])
    return np.array(results)


def random_replace(trg_seq, portion_replaced):
    size = int(trg_seq.size(0) * trg_seq.size(1) * portion_replaced)
    a = []
    b = []
    for i in range(trg_seq.size(0)):
        for j in range(trg_seq.size(1)):
            a.append(i)
            b.append(j)
    while len(a) > size:
        idx = random.choice(range(len(a)))
        del (a[idx])
        del (b[idx])
    trg_seq[a, b] = torch.from_numpy(np.random.randint(7, size=size)).long()
    return trg_seq


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


def main():
    vis = visdom.Visdom(env='sequence_tester')
    device_s = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    extensions = ['stan', 'cont', 'pred', 'noised', 'seq']
    extensions = ['noised']

    # Parameters
    # sequentiaol_model_name = 'transformer'
    sequentiaol_model_name = 'LSTM'
    video_names = ['video' + str(i).zfill(2) for i in range(1, 41)]
    # test_video = 1  # two videos in test set, write 0 or 1

    f1 = 0
    cm_all = np.zeros((7, 7))
    acc_avg = []
    portion = np.zeros((7,))
    for video_name in video_names:
        open_path = os.path.join('/home/yitong/venv_yitong/cholec80_phase/data/whole', video_name)

        file = open(os.path.join(open_path, 'seq_true.pickle'), 'rb')
        seq_true = pickle.load(file)
        file.close()

        seq_true = np.array(seq_true)
        phases, count = np.unique(seq_true, return_counts=True)
        count = count / (np.sum(count))

        for idx, phase in enumerate(phases):
            portion[phase] += count[idx]
    portion = portion/40
    print(portion)
    print(1/portion)


if __name__ == '__main__':
    main()
