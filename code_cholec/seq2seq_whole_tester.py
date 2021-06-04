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
    extensions = ['pred']

    # Parameters
    # sequentiaol_model_name = 'transformer'
    sequentiaol_model_name = 'LSTM'
    video_names = ['video' + str(i).zfill(2) for i in range(61, 81)]
    # test_video = 1  # two videos in test set, write 0 or 1

    f1 = 0
    for video_name in video_names:
        for extension in extensions:
            # extension = extensions[2]
            vis.close('Surgical workflow sequential')
            print('The sequential model chosen is: %s' % sequentiaol_model_name, '\n'
                  'The test video is', video_name, '\n'
                  'The extension is: %s' % extension)
            if sequentiaol_model_name == 'transformer':
                sequential_model = Transformer(trg_pad_idx=6, n_trg_vocab=7,
                                               d_word_vec=1200, d_model=1200, d_inner=1000,
                                               n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=100).to(device_s)
                # sequential_model.load_state_dict(torch.load('./params/params_trans_90_' +
                #                                             extension + '.pkl'))
                sequential_model.load_state_dict(torch.load('./params/params_trans_90.pkl'))
                slin_sz = 30
            else:
                sequential_model = seq2seq_LSTM(hidden_dim=1200, num_layers=3, tar_seq_dim=7).to(device_s)
                # sequential_model.load_state_dict(torch.load('./params/cross_validation/' + folder + '/params_LSTM_90_' +
                #                                             extension + '.pkl'))
                sequential_model.load_state_dict(torch.load('./params/params_LSTM_seq2seq_whole.pkl'))
                # sequential_model.load_state_dict(torch.load('./params/params_LSTM_save_point.pkl'))
                slin_sz = 17
            sequential_model.eval()

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
            # vis.line(X=np.array(range(len(seq_pre))), Y=np.column_stack((np.array(seq_pre), np.array(seq_true))),
            #          win='Surgical workflow', update='append',
            #          opts=dict(title='Surgical workflow', showlegend=True, legend=['Prediction', 'Ground Truth']))
            # print('confusion matrix:', cm)

            # mode average with a sliding window
            mode_av = mode_average(np.array(seq_pre), slinwin_sz=17)
            cm = confusion_matrix(np.array(seq_true), mode_av, labels=[0, 1, 2, 3, 4, 5])
            cm = cm / np.sum(cm, axis=1)[:, None] * 100
            anot = np.sum(np.diag(cm)[1:]) / 5
            print('[validation accuracy: %.3f %% accuracy without transition phase: %.3f%%'
                  % (np.sum(np.diag(cm)) / 6, anot))
            # vis.heatmap(X=cm, win='heatmap2', opts=dict(title='confusion matrix mode',
            #                                             rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
            #                                             columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
            # vis.line(X=np.array(range(len(seq_pre))), Y=np.column_stack((mode_av, np.array(seq_true))),
            #          win='Surgical workflow MODE', update='append',
            #          opts=dict(title='Surgical workflow MODE', showlegend=True, legend=['Prediction', 'Ground Truth']))

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

            # Apply the sequential model 90
            # make an array for all predicted sequence
            sequece_label = torch.zeros(len(slinwin_list) + 1, sequence_length)
            sequece_label[:, :] = torch.tensor(float('nan'))
            sequece_label[0, 0:90] = torch.tensor(mode_av.reshape(1, -1)[0, 0:90])
            # sequece_label[0, 0:90] = torch.tensor(np.array(seq_true).reshape(1, -1)[0, 0:90])
            for i, cur_slwin in enumerate(slinwin_list):
                sequence_input = torch.zeros(1, cur_slwin[-1]+1, 1200)
                trg_seq = sequece_label[i, 0:cur_slwin[-1]+1].unsqueeze(0).long()  # the current 90 clips
                # trg_seq = torch.tensor(np.array(seq_true).reshape(1, -1)
                #                        [0, cur_slwin[0]:cur_slwin[90]]).unsqueeze(0).long()
                # trg_seq = torch.randint(0, 7, (1, 90)).long().to(device_s)

                # introduce noise into the target sequence
                # trg_seq = random_replace(trg_seq, 0.9).long().to(device_s)
                trg_seq = trg_seq.to(device_s)

                for j in range(cur_slwin[-1]+1):
                    sequence_input[:, j, :] = fc_list[j]
                sequence_input = sequence_input.to(device_s)
                with torch.no_grad():
                    sequence_output = sequential_model.forward(sequence_input, trg_seq)
                _, predicted_labels = torch.max(sequence_output.cpu().data, 2)
                sequece_label[i + 1, cur_slwin[-10]:(cur_slwin[-1] + 1)] = predicted_labels[-10:]  # save the predictions to the next
                # print(predicted_labels)
                # row
            # sequece_label[0, :] = torch.tensor(float('nan'))
            sequece_label, _ = torch.mode(sequece_label, 0)
            # sequece_label= mode_average(sequece_label.numpy(), slinwin_sz=slin_sz)
            sequece_label = sequece_label.numpy()

            cm = confusion_matrix(np.array(seq_true), sequece_label, labels=[0, 1, 2, 3, 4, 5, 6])
            cm = cm / np.sum(cm, axis=1)[:, None] * 100
            cm_inner = cm[1:6, 1:6]
            cm_inner = cm_inner / np.sum(cm_inner, axis=1)
            anot = np.sum(np.diag(cm_inner)[1:]) / 5
            x = list(np.diag(cm_inner)[1:6])
            # txt3 = ''.join(['Phase %d accuracy:%d%%   ' % (i + 1, x[i]) for i in range(len(x))])
            # vis.text(txt3, win='cur_best', opts=dict(title='Current Best'))
            # print('[validation accuracy: %.3f %% accuracy without transition phase: %.3f%%'
            #       % (np.sum(np.diag(cm)) / 6, anot))
            # vis.heatmap(X=cm, win='heatmap3', opts=dict(title='confusion matrix sequential',
            #                                             rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
            #                                             columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
            f1 += phase_f1(np.array(seq_true), sequece_label)

            # vis.line(X=np.array(range(len(seq_pre))), Y=np.column_stack((sequece_label, np.array(seq_true), np.array(seq_pre))),
            #          win='Surgical workflow sequential', update='append',
            #          opts=dict(title='Surgical workflow sequential', showlegend=True,
            #                    legend=['Prediction', 'Ground Truth', 'c3d'],
            #                    width=800, height=200))
            vis.line(X=np.array(range(len(seq_pre))), Y=np.column_stack((sequece_label, np.array(seq_true))),
                     win='Surgical workflow sequential', update='append',
                     opts=dict(title='Surgical workflow sequential', showlegend=True,
                               legend=['Prediction', 'Ground Truth'],
                               width=800, height=200))

            # save_path = os.path.join('/home/yitong/venv_yitong/cholec80_phase/results', video_name)
            # if not os.path.exists(save_path):
            #     os.mkdir(save_path)
            #
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
            #
            # file = open(os.path.join(save_path, 'seq_' + sequentiaol_model_name + '_90_' + extension + '.pickle'), 'wb')
            # pickle.dump(list(sequece_label), file)
            # file.close()

    print(f1/20)

if __name__ == '__main__':
    main()
