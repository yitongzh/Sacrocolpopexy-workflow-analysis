import sys
from model import Endo3D
from many2many_LSTM import many2many_LSTM
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



def main():
    vis = visdom.Visdom(env='sequence_tester')
    device_s = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    extensions = ['stan', 'cont', 'pred', 'noised', 'seq']

    # Parameters
    # sequentiaol_model_name = 'transformer'
    sequentiaol_model_name = 'LSTM'

    sequential_model = many2many_LSTM(hidden_dim=1200, num_layers=3).to(device_s)
    sequential_model.load_state_dict(torch.load('./params/params_LSTM_m2m.pkl'))
    video_names = ['video' + str(i).zfill(2) for i in range(61, 81)]

    for video_name in video_names:
        print('The sequential model chosen is: %s' % sequentiaol_model_name, '\n'
              'The test video is:', video_name)

        open_path = os.path.join('/home/yitong/venv_yitong/cholec80_phase/data/whole', video_name)

        file = open(os.path.join(open_path, 'seq_pred.pickle'), 'rb')
        seq_pre = pickle.load(file)
        file.close()

        file = open(os.path.join(open_path, 'seq_true.pickle'), 'rb')
        seq_true = pickle.load(file)
        file.close()

        file = open(os.path.join(open_path, 'fc_list.pickle'), 'rb')
        fc_list = pickle.load(file)
        file.close()

        # vis.heatmap(X=cm, win='heatmap', opts=dict(title='confusion matrix',
        #                                            rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
        #                                            columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
        vis.line(X=np.array(range(len(seq_pre))), Y=np.column_stack((np.array(seq_pre), np.array(seq_true))),
                 win='Surgical workflow', update='append',
                 opts=dict(title='Surgical workflow', showlegend=True, legend=['Prediction', 'Ground Truth']))
        # print('confusion matrix:', cm)

        # mode average with a sliding window
        mode_av = mode_average(np.array(seq_pre), slinwin_sz=17)
        cm = confusion_matrix(np.array(seq_true), mode_av, labels=[0, 1, 2, 3, 4, 5])
        cm = cm / np.sum(cm, axis=1) * 100
        anot = np.sum(np.diag(cm)[1:]) / 5
        print('[validation accuracy: %.3f %% accuracy without transition phase: %.3f%%'
              % (np.sum(np.diag(cm)) / 6, anot))
        vis.heatmap(X=cm, win='heatmap2', opts=dict(title='confusion matrix mode',
                                                    rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
                                                    columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
        vis.line(X=np.array(range(len(seq_pre))), Y=np.column_stack((mode_av, np.array(seq_true))),
                 win='Surgical workflow MODE', update='append',
                 opts=dict(title='Surgical workflow MODE', showlegend=True, legend=['Prediction', 'Ground Truth']))

        # make a list that contains all the index for all sequences
        index = list(range(len(fc_list)))
        slwin_size = 100
        step = 10
        start_index = 0
        sequence_length = len(fc_list)
        slinwin_list = []
        while start_index + slwin_size <= len(index):
            cur_slwin = index[start_index: start_index + slwin_size]
            slinwin_list.append(cur_slwin)
            start_index += step
        if start_index + slwin_size - step != len(index):
            cur_slwin = index[-slwin_size:]
            slinwin_list.append(cur_slwin)

        # Apply the sequential model 100
        sequece_label = torch.zeros(len(slinwin_list) + 1, sequence_length)
        sequece_label[:, :] = torch.tensor(float('nan'))
        # sequece_label[0, :] = torch.tensor(mode_av.reshape(1, -1))
        sequece_label[0, :] = torch.tensor(np.array(seq_pre).reshape(1, -1))
        for i, cur_slwin in enumerate(slinwin_list):
            sequence_input = torch.zeros(1, slwin_size, 1200)
            for j, idx in enumerate(cur_slwin):
                sequence_input[:, j, :] = fc_list[idx]
            sequence_input = sequence_input.to(device_s)
            sequence_output = sequential_model.forward(sequence_input)
            _, predicted_labels = torch.max(sequence_output.cpu().data, 2)
            sequece_label[i + 1, cur_slwin[0]:(cur_slwin[-1] + 1)] = predicted_labels
        sequece_label[0, :] = torch.tensor(float('nan'))
        sequece_label, _ = torch.mode(sequece_label, 0)
        # mode_av = mode_average(sequece_label.numpy(), slinwin_sz=slin_sz)
        sequece_label = sequece_label.numpy()

        vis.line(X=np.array(range(len(seq_pre))), Y=np.column_stack((sequece_label, np.array(seq_true))),
                 win='Surgical workflow sequential', update='append',
                 opts=dict(title='Surgical workflow sequential', showlegend=True, legend=['Prediction', 'Ground Truth']))

        # cm = confusion_matrix(np.array(seq_true), sequece_label, labels=[0, 1, 2, 3, 4, 5, 6])
        # cm = cm / np.sum(cm, axis=1) * 100
        # cm_inner = cm[1:6, 1:6]
        # cm_inner = cm_inner / np.sum(cm_inner, axis=1)
        # anot = np.sum(np.diag(cm_inner)[1:]) / 5
        # x = list(np.diag(cm_inner)[1:6])
        # txt3 = ''.join(['Phase %d accuracy:%d%%   ' % (i + 1, x[i]) for i in range(len(x))])
        # vis.text(txt3, win='cur_best', opts=dict(title='Current Best'))
        # print('[validation accuracy: %.3f %% accuracy without transition phase: %.3f%%'
        #       % (np.sum(np.diag(cm)) / 6, anot))
        # # vis.heatmap(X=cm, win='heatmap3', opts=dict(title='confusion matrix sequential',
        # #                                             rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
        # #                                             columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
        # vis.line(X=np.array(range(len(seq_pre))), Y=np.column_stack((sequece_label, np.array(seq_true))),
        #          win='Surgical workflow sequential', update='append',
        #          opts=dict(title='Surgical workflow sequential', showlegend=True, legend=['Prediction', 'Ground Truth']))

        save_path = os.path.join('/home/yitong/venv_yitong/cholec80_phase/results', video_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # file = open(os.path.join(save_path, 'seq_true.pickle'), 'wb')
        # pickle.dump(list(seq_true), file)
        # file.close()
        #
        # file = open(os.path.join(save_path, 'seq_pred.pickle'), 'wb')
        # pickle.dump(list(seq_pre), file)
        # file.close()
        #
        # file = open(os.path.join(save_path, 'seq_mode_av.pickle'), 'wb')
        # pickle.dump(list(mode_av), file)
        # file.close()

        file = open(os.path.join(save_path, 'seq_' + sequentiaol_model_name + '_100_m2m.pickle'), 'wb')
        pickle.dump(list(sequece_label), file)
        file.close()


if __name__ == '__main__':
    main()