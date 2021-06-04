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

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
from torch.utils import data
import pickle
import json
import random

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


def mode_average_1(prediction, slinwin_sz=17):
    results = []
    hf = int((slinwin_sz - 1) / 2)
    prediction_pad = np.hstack((np.full([hf, ], np.nan), prediction))
    prediction_pad = np.hstack((prediction_pad, np.full([hf, ], np.nan)))
    for i in range(prediction.shape[0]):
        st_idx = i
        ed_idx = i + slinwin_sz - 1
        results.append(stats.mode(prediction_pad[st_idx:ed_idx])[0][0])
        prediction_pad[i + hf] = stats.mode(prediction_pad[st_idx:ed_idx])[0][0]
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

    correct_pred_eval = (torch.tensor(seq_pred) == torch.tensor(seq_true)).sum().item()
    total_pred_eval = torch.tensor(seq_true).size(0)
    acc = correct_pred_eval / total_pred_eval

    phases, weight = np.unique(seq_true, return_counts=True)
    weight = weight/sum(weight)
    weight = dict(zip(phases, weight))
    f1s = {}
    precisions = {}
    recalls = {}
    f1_micro = 0
    for phase in range(1, 6):
        if phase in phases:
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
            if tp != 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
            else:
                precision = 0
                recall = 0

            f1_micro += f1 * weight[phase]
            f1s.update({phase: f1})
            precisions.update({phase: precision})
            recalls.update({phase: recall})
        else:
            f1s.update({phase: np.NaN})
            precisions.update({phase: np.NaN})
            recalls.update({phase: np.NaN})
    return {'f1': f1s, 'precision': precisions, 'recall': recalls,
            'f1_avg': sum(f1s) / len(f1s),
            'f1_micro': f1_micro, 'acc_micro': acc,
            'precision_avg': sum(precisions) / len(precisions),
            'recall_avg': sum(recalls) / len(recalls)}


def main():
    vis = visdom.Visdom(env='sequence_tester')
    device_s = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    extensions = ['stan', 'cont', 'pred', 'noised', 'seq']
    extensions = ['stan']

    # Parameters
    # sequentiaol_model_name = 'transformer'
    sequentiaol_model_name = 'LSTM'
    video_names = ['video' + str(i).zfill(2) for i in range(61, 81)]

    f1 = 0
    cm_all = np.zeros((7, 7))
    f1_micro = []
    acc_micro = []
    detail_results_list = []
    for video_name in video_names:
        for extension in extensions:
            # extension = extensions[2]
            print('The sequential model chosen is: %s' % sequentiaol_model_name, '\n'
                  'The test video is:', video_name)
            if sequentiaol_model_name == 'transformer':
                sequential_model = Transformer(trg_pad_idx=6, n_trg_vocab=7,
                                               d_word_vec=1200, d_model=1200, d_inner=1000,
                                               n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=100).to(
                    device_s)
                sequential_model.load_state_dict(
                    torch.load('./params/params_trans_100_' + extension + '.pkl'))
                # sequential_model.load_state_dict(torch.load('./params/params_trans_save_point.pkl'))
                slin_sz = 40
            else:
                sequential_model = seq2seq_LSTM(hidden_dim=1200, num_layers=3).to(device_s)
                # sequential_model.load_state_dict(
                #     torch.load('./params/params_LSTM_100_' + extension + '.pkl'))
                sequential_model.load_state_dict(torch.load('./params/params_LSTM_100_noised_w.pkl'))
                slin_sz = 40

            open_path = os.path.join('/home/yitong/venv_yitong/cholec80_phase/data/whole', video_name)
            vis.close('Surgical workflow sequential')

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
            # vis.line(X=np.array(range(len(seq_true))), Y=np.column_stack((seq_pre, np.array(seq_true))),
            #          win='Surgical workflow', update='append',
            #          opts=dict(title='Surgical workflow', showlegend=True, legend=['Prediction', 'Ground Truth']))
            # print('confusion matrix:', cm)

            # mode average with a sliding window
            mode_av = mode_average(np.array(seq_pre), slinwin_sz=17)
            # mode_av = seq_pre
            cm = confusion_matrix(np.array(seq_true), mode_av, labels=[0, 1, 2, 3, 4, 5])
            cm = cm / np.sum(cm, axis=1) * 100
            anot = np.sum(np.diag(cm)[1:]) / 5
            print('[validation accuracy: %.3f %% accuracy without transition phase: %.3f%%'
                  % (np.sum(np.diag(cm)) / 6, anot))
            # vis.heatmap(X=cm, win='heatmap2', opts=dict(title='confusion matrix mode',
            #                                             rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
            #                                             columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
            # vis.line(X=np.array(range(len(seq_true))), Y=np.column_stack((mode_av, np.array(seq_true))),
            #          win='Surgical workflow MODE', update='append',
            #          opts=dict(title='Surgical workflow MODE', showlegend=True, legend=['Prediction', 'Ground Truth']))

            # make a list that contains all the index for all sequences
            index = list(range(len(fc_list)))
            slwin_size = 100
            step = 100
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
                # trg_seq = sequece_label[0, cur_slwin[0]:cur_slwin[-1] + 1].unsqueeze(0).long().to(device_s)
                trg_seq = random_replace(torch.tensor(seq_true[cur_slwin[0]:cur_slwin[-1] + 1]).unsqueeze(0)
                                         , 0.7).long().to(device_s)
                for j, idx in enumerate(cur_slwin):
                    sequence_input[:, j, :] = fc_list[idx]
                sequence_input = sequence_input.to(device_s)
                sequence_output = sequential_model.forward(sequence_input, trg_seq)
                _, predicted_labels = torch.max(sequence_output.cpu().data, 2)
                sequece_label[i + 1, cur_slwin[0]:(cur_slwin[-1] + 1)] = predicted_labels
            sequece_label[0, :] = torch.tensor(float('nan'))
            sequece_label, _ = torch.mode(sequece_label, 0)
            # mode_av = mode_average(sequece_label.numpy(), slinwin_sz=slin_sz)
            sequece_label = sequece_label.numpy()

            cm = confusion_matrix(np.array(seq_true), sequece_label, labels=[0, 1, 2, 3, 4, 5, 6])
            cm_all += cm
            # txt3 = ''.join(['Phase %d accuracy:%d%%   ' % (i + 1, x[i]) for i in range(len(x))])
            # vis.text(txt3, win='cur_best', opts=dict(title='Current Best'))
            # print('[validation accuracy: %.3f %% accuracy without transition phase: %.3f%%'
            #       % (np.sum(np.diag(cm)) / 6, anot))
            # vis.heatmap(X=cm, win='heatmap3', opts=dict(title='confusion matrix sequential',
            #                                             rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
            #                                             columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))

            results = phase_f1(np.array(seq_true), sequece_label)
            acc_micro.append(results['acc_micro'])
            detail_results_list.append(results)


            # sequece_label = torch.zeros(len(slinwin_list) + 1, sequence_length)
            # sequece_label[:, :] = torch.tensor(float('nan'))
            # sequece_label[0, :] = torch.tensor(mode_av.reshape(1, -1))
            # for i, cur_slwin in enumerate(slinwin_list):
            #     sequence_input = torch.zeros(1, slwin_size, 1200)
            #     trg_seq = sequece_label[0, cur_slwin[0]:cur_slwin[-1] + 1].unsqueeze(0).long().to(device_s)
            #     for j, idx in enumerate(cur_slwin):
            #         sequence_input[:, j, :] = fc_list[idx]
            #     sequence_input = sequence_input.to(device_s)
            #     sequence_output = sequential_model.forward(sequence_input, trg_seq)
            #     _, predicted_labels = torch.max(sequence_output.cpu().data, 2)
            #     sequece_label[i + 1, cur_slwin[0]:(cur_slwin[-1] + 1)] = predicted_labels
            # sequece_label[0, :] = torch.tensor(float('nan'))
            # sequece_label, _ = torch.mode(sequece_label, 0)
            # mode_av = mode_average(sequece_label.numpy(), slinwin_sz=slin_sz)
            vis.line(X=np.array(range(len(seq_true))), Y=np.column_stack((sequece_label, np.array(seq_true))),
                     win='Surgical workflow sequential', update='append',
                     opts=dict(title='Surgical workflow sequential', showlegend=True,
                               legend=['Prediction', 'Ground Truth']))

            # cm = confusion_matrix(np.array(seq_true), mode_av, labels=[0, 1, 2, 3, 4, 5])
            # # cm = cm / np.sum(cm, axis=1) * 100
            # cm_inner = cm[1:6, 1:6]
            # cm_inner = cm_inner / np.sum(cm_inner, axis=1)
            # print(cm_inner)
            # anot = np.sum(np.diag(cm_inner)[1:]) / 5
            # x = list(np.diag(cm_inner))
            # txt3 = ''.join(['Phase %d accuracy:%d%%   ' % (i + 1, x[i]) for i in range(len(x))])
            # vis.text(txt3, win='cur_best', opts=dict(title='Current Best'))

            # print('[validation accuracy: %.3f %% accuracy without transition phase: %.3f%%'
            #       % (np.sum(np.diag(cm[0:6, 0:6])) / 6, anot))
            # vis.heatmap(X=cm_inner, win='heatmap3', opts=dict(title='confusion matrix sequential',
            #                                                   rownames=['t1', 't2', 't3', 't4', 't5'],
            #                                                   columnnames=['p1', 'p2', 'p3', 'p4', 'p5']))

            # validation_input = open('/home/yitong/venv_yitong/sacro_wf_analysis/temp/tar_seq.pickle', 'wb')
            # pickle.dump(list(mode_av), validation_input)
            # validation_input.close()

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

            # file = open(os.path.join(save_path, 'seq_' + sequentiaol_model_name + '_100_' + extension + '.pickle'),
            #             'wb')
            # pickle.dump(list(sequece_label), file)
            # file.close()
    cm_all = cm_all / np.sum(cm_all, axis=1)[:, None]
    vis.heatmap(X=cm_all, win='heatmap', opts=dict(title='confusion matrix',
                                               rownames=['t0', 't1', 't2', 't3', 't4', 't5', 't_not'],
                                               columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p_not']))
    detail_precision = np.zeros((len(detail_results_list), 5))
    detail_recall = np.zeros((len(detail_results_list), 5))
    accuracy_std = np.std(np.array(acc_micro))
    print('accuracy:', sum(acc_micro)/len(acc_micro), accuracy_std)
    for idx, item in enumerate(detail_results_list):
        detail_precision[idx, :] = np.array(np.array(list(item['precision'].values())))
        detail_recall[idx, :] = np.array(np.array(list(item['recall'].values())))
    precision_std = np.std(np.nanmean(detail_precision, axis=1))
    recall_std = np.std(np.nanmean(detail_recall, axis=1))
    print('precisions:', np.nanmean(detail_precision), precision_std)
    print('recalls:', np.nanmean(detail_recall), recall_std)


if __name__ == '__main__':
    main()
