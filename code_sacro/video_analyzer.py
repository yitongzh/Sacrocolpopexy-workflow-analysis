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
    extension = extensions[4]
    # model_type = '_90_'
    model_type = '_100_'
    video_name = 'becb53be-978f-4233-bbb1-ed854f48dc21'
    # video_name = 'd6994bf0-b53c-49c6-8043-c485fd847e4a'

    # Parameters
    sequentiaol_model_name = 'transformer'
    # sequentiaol_model_name = 'LSTM'

    # extension = extensions[2]
    print('The sequential model chosen is: %s' % sequentiaol_model_name, '\n'
          'The extension is: %s' % extension)
    if model_type == '_90_':
        if sequentiaol_model_name == 'transformer':
            sequential_model = Transformer(trg_pad_idx=6, n_trg_vocab=7,
                                           d_word_vec=1200, d_model=1200, d_inner=1000,
                                           n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=100).to(device_s)
            sequential_model.load_state_dict(torch.load('./params/cross_validation/div3/params_trans_90_' +
                                                        extension + '.pkl'))
            # sequential_model.load_state_dict(torch.load('./params/params_trans_save_point.pkl'))
            slin_sz = 30
        else:
            sequential_model = seq2seq_LSTM(hidden_dim=1200, num_layers=3).to(device_s)
            sequential_model.load_state_dict(torch.load('./params/cross_validation/div3/params_LSTM_90_' +
                                                        extension + '.pkl'))
            # sequential_model.load_state_dict(torch.load('./params/params_LSTM_save_point.pkl'))
            slin_sz = 17
    elif model_type == '_100_':
        if sequentiaol_model_name == 'transformer':
            sequential_model = Transformer(trg_pad_idx=6, n_trg_vocab=7,
                                           d_word_vec=1200, d_model=1200, d_inner=1000,
                                           n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=100).to(
                device_s)
            sequential_model.load_state_dict(
                torch.load('./params/cross_validation/div3/params_trans_100_' + extension + '.pkl'))
            slin_sz = 40
        else:
            sequential_model = seq2seq_LSTM(hidden_dim=1200, num_layers=3).to(device_s)
            sequential_model.load_state_dict(
                torch.load('./params/cross_validation/div3/params_LSTM_100_' +
                           extension + '.pkl'))
            slin_sz = 40

    open_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/data/'
                             'sacro_sequence/whole/test/' + video_name)

    file = open(os.path.join(open_path, 'seq_pred.pickle'), 'rb')
    seq_pre = pickle.load(file)
    file.close()

    file = open(os.path.join(open_path, 'fc_list.pickle'), 'rb')
    fc_list = pickle.load(file)
    file.close()

    # mode average with a sliding window
    mode_av = mode_average(np.array(seq_pre), slinwin_sz=17)

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

    if model_type == '_90_':
        # Apply the sequential model 90
        # make an array for all predicted sequence
        sequece_label = torch.zeros(len(slinwin_list) + 1, sequence_length)
        sequece_label[:, :] = torch.tensor(float('nan'))
        sequece_label[0, 0:90] = torch.tensor(mode_av.reshape(1, -1)[0, 0:90])
        # sequece_label[0, :] = torch.tensor(np.array(seq_true).reshape(1, -1)[0, 0:90])
        for i, cur_slwin in enumerate(slinwin_list):
            sequence_input = torch.zeros(1, slwin_size, 1200)
            trg_seq = sequece_label[i, cur_slwin[0]:cur_slwin[90]].unsqueeze(0).long()  # the current 90 clips
            # trg_seq = torch.tensor(np.array(seq_true).reshape(1, -1)
            #                        [0, cur_slwin[0]:cur_slwin[90]]).unsqueeze(0).long()
            # trg_seq = torch.randint(0, 7, (1, 90)).long().to(device_s)

            # introduce noise into the target sequence
            # trg_seq = random_replace(trg_seq, 0.9).long().to(device_s)
            trg_seq = trg_seq.to(device_s)

            for j, idx in enumerate(cur_slwin):
                sequence_input[:, j, :] = fc_list[idx]
            sequence_input = sequence_input.to(device_s)
            sequence_output = sequential_model.forward(sequence_input, trg_seq)
            _, predicted_labels = torch.max(sequence_output.cpu().data, 2)
            sequece_label[i + 1, cur_slwin[10]:(cur_slwin[-1] + 1)] = predicted_labels  # save the predictions to the next
            # print(predicted_labels)
            # row
        # sequece_label[0, :] = torch.tensor(float('nan'))
        sequece_label, _ = torch.mode(sequece_label, 0)
        # sequece_label= mode_average(sequece_label.numpy(), slinwin_sz=slin_sz)
        sequece_label = sequece_label.numpy()

    elif model_type == '_100_':
        # Apply the sequential model 100
        sequece_label = torch.zeros(len(slinwin_list) + 1, sequence_length)
        sequece_label[:, :] = torch.tensor(float('nan'))
        # sequece_label[0, :] = torch.tensor(mode_av.reshape(1, -1))
        sequece_label[0, :] = torch.tensor(np.array(seq_pre).reshape(1, -1))
        for i, cur_slwin in enumerate(slinwin_list):
            sequence_input = torch.zeros(1, slwin_size, 1200)
            trg_seq = sequece_label[0, cur_slwin[0]:cur_slwin[-1] + 1].unsqueeze(0).long().to(device_s)
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

    save_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/results/test/' + video_name)

    file = open(os.path.join(save_path, 'seq_pred.pickle'), 'wb')
    pickle.dump(list(seq_pre), file)
    file.close()

    file = open(os.path.join(save_path, 'seq_mode_av.pickle'), 'wb')
    pickle.dump(list(mode_av), file)
    file.close()

    file = open(os.path.join(save_path, 'seq_' + sequentiaol_model_name + model_type + extension + '.pickle'), 'wb')
    pickle.dump(list(sequece_label), file)
    file.close()


if __name__ == '__main__':
    main()
