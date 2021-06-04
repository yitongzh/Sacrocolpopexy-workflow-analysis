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

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
from torch.utils import data
import pickle

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


def main():
    vis = visdom.Visdom(env='sequence_tester')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_s = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = Endo3D().to(device)

    sequentiaol_model_name = 'transformer'
    sequentiaol_model_name = 'LSTM'
    folder = 'div4'
    print('The sequential model chosen is: %s' % sequentiaol_model_name)
    if sequentiaol_model_name == 'transformer':
        sequential_model = Transformer(trg_pad_idx=6, n_trg_vocab=7,
                                       d_word_vec=1200, d_model=1200, d_inner=1000,
                                       n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=100).to(device_s)
        sequential_model.load_state_dict(torch.load('./params/cross_validation/' + folder + '/params_trans_75.pkl'))
        # sequential_model.load_state_dict(torch.load('./params/params_trans_save_point.pkl'))
        slin_sz = 30
    else:
        sequential_model = seq2seq_LSTM(hidden_dim=1200, num_layers=3).to(device_s)
        sequential_model.load_state_dict(torch.load('./params/cross_validation/' + folder + '/params_LSTM_75.pkl'))
        # sequential_model.load_state_dict(torch.load('./params/params_LSTM_save_point.pkl'))
        slin_sz = 17

    # load pre-trained parameters
    # model.load_state_dict(torch.load('./params/params_endo3d.pkl'))
    model.load_state_dict(torch.load('./params/cross_validation/div1/params_endo3d.pkl'))
    # model1.load_state_dict(torch.load('./params/params_endo3d_1vo2.pkl'))

    # loading the training and validation set
    print('loading data')
    start = time.time()
    # sacro = sacro_loder(batch_num=10, validation_batch_size=500)
    sacro = sequence_loder()

    video_name = 'a90983a1-2329-4019-96e8-949493c3fc24'  # div4
    # video_name = '26d9e19b-59ed-4bbb-906a-9953b5d2c825'  # div4
    # video_name = 'e6a9f8f7-8024-42d5-aad5-ef726d353f3b'  # div4
    # video_name = '5238ac7b-ad85-488f-b013-9ee4e2064e60'  # div1

    # video_name = '436c714b-b588-4894-b68c-1e0f773e5df1'
    # video_name = 'e8d3eabc-edd1-414f-9d95-277df742655a'
    video_name = '315ac662-9e95-4dec-80f0-48e244f9f8e1'

    sacro.whole_len_output(video_name)
    whole_loder = data.DataLoader(sacro, 20)
    elapsed = (time.time() - start)
    print("Data loded, time used:", elapsed)

    # Initializing necessary components
    loss_func = nn.NLLLoss().to(device)

    # Evaluation on validation set
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    valid_num = 0
    cm = np.zeros((6, 6))
    seq_pre = []
    seq_true = []
    fc_list = []
    for labels_val, inputs_val in tqdm(whole_loder, ncols=80):
        inputs_val = inputs_val.float().to(device)
        labels_val = labels_val.long().to(device)

        output, x_fc = model.forward_cov(inputs_val)
        valid_loss = loss_func(output, labels_val)
        # Calculate the loss with new parameters
        running_loss += valid_loss.item()
        # current_loss = running_loss / (batch_counter + 1)

        _, predicted_labels = torch.max(output.cpu().data, 1)
        cm += confusion_matrix(labels_val.cpu().numpy(), predicted_labels, labels=[0, 1, 2, 3, 4, 5])
        correct_pred = (predicted_labels == labels_val.cpu()).sum().item()
        for i in range(predicted_labels.numpy().shape[0]):
            seq_pre.append(predicted_labels.numpy()[i])
            seq_true.append(labels_val.cpu().numpy()[i])
            fc_list.append(x_fc[i, :])
        total_pred = predicted_labels.size(0)
        accuracy = correct_pred / total_pred
        running_accuracy += accuracy
        valid_num += 1
        # current_accuracy = running_accuracy / (batch_counter + 1) * 100
    batch_loss = running_loss / valid_num
    batch_accuracy = running_accuracy / valid_num * 100
    cm = cm / np.sum(cm, axis=1) * 100
    anot = np.sum(np.diag(cm)[1:]) / 5
    print('[validation loss: %.3f validation accuracy: %.3f %% accuracy without transition phase: %.3f%%'
          % (batch_loss, batch_accuracy, anot))
    vis.heatmap(X=cm, win='heatmap', opts=dict(title='confusion matrix',
                                               rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
                                               columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
    vis.line(X=np.array(range(len(sacro))), Y=np.column_stack((np.array(seq_pre), np.array(seq_true))),
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
    vis.line(X=np.array(range(len(sacro))), Y=np.column_stack((mode_av, np.array(seq_true))),
             win='Surgical workflow MODE', update='append',
             opts=dict(title='Surgical workflow MODE', showlegend=True, legend=['Prediction', 'Ground Truth']))

    # make a list that contains all the index for all sequences
    index = list(range(len(fc_list)))
    slwin_size = 100
    step = 25
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

    # Apply the sequential model 75
    sequece_label = torch.zeros(len(slinwin_list) + 1, sequence_length)
    sequece_label[:, :] = torch.tensor(float('nan'))
    sequece_label[0, 0:75] = torch.tensor(mode_av.reshape(1, -1)[0, 0:75])
    for i, cur_slwin in enumerate(slinwin_list):
        sequence_input = torch.zeros(1, slwin_size, 1200)
        trg_seq = sequece_label[i, cur_slwin[0]:cur_slwin[75]].unsqueeze(0).long().to(device_s)
        # trg_seq = torch.tensor(mode_av.reshape(1, -1)[0, cur_slwin[0]:cur_slwin[75]]).unsqueeze(0).long().to(device_s)
        for j, idx in enumerate(cur_slwin):
            sequence_input[:, j, :] = fc_list[idx]
        sequence_input = sequence_input.to(device_s)
        sequence_output = sequential_model.forward(sequence_input, trg_seq)
        _, predicted_labels = torch.max(sequence_output.cpu().data, 2)
        sequece_label[i + 1, cur_slwin[25]:(cur_slwin[-1] + 1)] = predicted_labels
    sequece_label[0, :] = torch.tensor(float('nan'))
    sequece_label, _ = torch.mode(sequece_label, 0)
    # sequece_label= mode_average(sequece_label.numpy(), slinwin_sz=slin_sz)
    sequece_label = sequece_label.numpy()

    vis.line(X=np.array(range(len(sacro))), Y=np.column_stack((sequece_label, np.array(seq_true))),
             win='Surgical workflow sequential', update='append',
             opts=dict(title='Surgical workflow sequential', showlegend=True, legend=['Prediction', 'Ground Truth']))

    cm = confusion_matrix(np.array(seq_true), sequece_label, labels=[0, 1, 2, 3, 4, 5, 6])
    # cm = cm / np.sum(cm, axis=1) * 100
    cm_inner = cm[1:6, 1:6]
    cm_inner = cm_inner / np.sum(cm_inner, axis=1)
    anot = np.sum(np.diag(cm_inner)[1:]) / 5
    x = list(np.diag(cm_inner)[1:6])
    txt3 = ''.join(['Phase %d accuracy:%d%%   ' % (i + 1, x[i]) for i in range(len(x))])
    vis.text(txt3, win='cur_best', opts=dict(title='Current Best'))
    print('[validation accuracy: %.3f %% accuracy without transition phase: %.3f%%'
          % (np.sum(np.diag(cm)) / 6, anot))
    vis.heatmap(X=cm, win='heatmap3', opts=dict(title='confusion matrix sequential',
                                                rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
                                                columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
    vis.line(X=np.array(range(len(sacro))), Y=np.column_stack((sequece_label, np.array(seq_true))),
             win='Surgical workflow sequential', update='append',
             opts=dict(title='Surgical workflow sequential', showlegend=True, legend=['Prediction', 'Ground Truth']))


    # validation_input = open('/home/yitong/venv_yitong/sacro_wf_analysis/temp/tar_seq.pickle', 'wb')
    # pickle.dump(list(mode_av), validation_input)
    # validation_input.close()


if __name__ == '__main__':
    main()
