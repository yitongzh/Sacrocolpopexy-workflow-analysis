#! /usr/bin/env python
import numpy as np

import pickle
import sys, time
import os
import json
import codecs
import visdom
import random

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim

from tqdm import tqdm
from model import Endo3D
from seq2seq_LSTM import seq2seq_LSTM
from transformer.transformer import Transformer
from utils.clip_sequence_loder import clip_sequence_loader
from sklearn.metrics import confusion_matrix


def main():
    # class end2end(nn.Module):
    #     def __init__(self, model2, insight_length=100):
    #         super(end2end, self).__init__()
    #         self.insight_length = insight_length
    #         self.model1 = Endo3D()
    #         self.model2 = model2
    #
    #     def forward(self, x, tar_seq):
    #         # batch_size = tar_seq.size(0)
    #         fc8s = self.model1.forward_cov(x[0].cuda())[1].unsqueeze(1)
    #         for i in range(self.insight_length-1):
    #             fc8s = torch.cat((fc8s, self.model1.forward_cov(x[i + 1].cuda())[1].unsqueeze(1)), 1)
    #         output_ = self.model2.forward(fc8s, tar_seq.cuda())
    #         return output_
    class end2end(nn.Module):
        def __init__(self, model2):
            super(end2end, self).__init__()
            self.model1 = Endo3D()
            self.model2 = model2

        def forward(self, x, tar_seq):
            batch_size = tar_seq.size(0)
            x1 = torch.zeros(batch_size, 100, 1200).cuda()
            for i in range(100):
                _, fc8 = self.model1.forward_cov(x[i].cuda())
                # fc8.retain_grad()
                x1[:, i, :] = fc8.clone()
            # x1.retain_grad()
            output_ = self.model2.forward(x1, tar_seq.cuda())
            return output_

    folders = ['folder1', 'folder2', 'folder3', 'folder4', 'folder5', 'folder6', 'folder7']
    divs = ['div1', 'div2', 'div3', 'div4', 'div5', 'div6', 'div7']
    counter = 2
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    sequential_model = seq2seq_LSTM(hidden_dim=1200, num_layers=3)
    model = end2end(sequential_model)
    # model.load_state_dict(
    #     torch.load('./params/cross_validation/' + divs[counter] + '/params_LSTM_90_end2end.pkl'))
    model.model1.load_state_dict(torch.load('./params/cross_validation/' + divs[counter] + '/params_endo3d.pkl'))
    model.model2.load_state_dict(torch.load('./params/cross_validation/' + divs[counter] + '/params_LSTM_90_seq.pkl'))
    model = nn.DataParallel(model)
    model = model.cuda()

    current_path = os.path.abspath(os.getcwd())
    videos_path = os.path.join(current_path, 'data/sacro_jpg')
    with open(os.path.join(videos_path, 'dataset_' + divs[counter] + '.json'), 'r') as json_data:
        temp = json.load(json_data)
    test_list = temp['test']
    sacro_test = clip_sequence_loader(test_list, is_augmentation=False)

    # Evaluate the model on the validation set
    cm = np.zeros((7, 7))
    running_loss = 0.0
    running_accuracy = 0.0
    for i in tqdm(range(len(sacro_test)), ncols=80):
        x_vali = sacro_test[i]
        inputs_vali = x_vali['inputs']
        tar_seq_vali = x_vali['labels'][:, 0:90]
        labels_vali = x_vali['labels'][:, 10:]

        with torch.no_grad():
            output_vali = model(inputs_vali, tar_seq_vali)
        _, predicted_labels = torch.max(output_vali.cpu().data, 2)
        cm += confusion_matrix(labels_vali.numpy().reshape(-1, ), predicted_labels.view(-1, ),
                               labels=[0, 1, 2, 3, 4, 5, 6])
        correct_pred = (predicted_labels == labels_vali).sum().item()
        total_pred = predicted_labels.size(0) * predicted_labels.size(1)
        running_accuracy += correct_pred / total_pred
        torch.cuda.empty_cache()
    average_accuracy = running_accuracy / len(sacro_test) * 100
    print(average_accuracy)
    print(cm / np.sum(cm, axis=1)[:, None] * 100)


    # sequece_label_a = torch.zeros(len(sacro_test) + 1, sacro_test.sliwins[test_list[0]][-1][-1])
    # sequece_label_a[:, :] = torch.tensor(float('nan'))
    # sequece_label_a[0, 0:90] = sacro_test[0]['labels'][0, 0:90]
    #
    # sequece_label_b = torch.zeros(len(sacro_test) + 1, sacro_test.sliwins[test_list[1]][-1][-1])
    # sequece_label_b[:, :] = torch.tensor(float('nan'))
    # sequece_label_b[0, 0:90] = sacro_test[0]['labels'][1, 0:90]

    # cm = np.zeros((7, 7))
    # running_loss = 0.0
    # running_accuracy = 0.0
    # for i in tqdm(range(len(sacro_test)), ncols=80):
    #     x_test = sacro_test[i]
    #     inputs_vali = x_test['inputs']
    #     labels_vali = x_test['labels'][:, 10:]
    #
    #     sidx_a = sacro_test.sliwins[test_list[0]][i][0]
    #     sidx_b = sacro_test.sliwins[test_list[1]][i][0]
    #     eidx_a = sacro_test.sliwins[test_list[0]][i][-1]
    #     eidx_b = sacro_test.sliwins[test_list[1]][i][-1]
    #     tar_seq_vali = torch.cat((sequece_label_a[i, sidx_a:eidx_a].unsqueeze(0), sequece_label_b[i, sidx_b:eidx_b].unsqueeze(0)), 0)
    #
    #     with torch.no_grad():
    #         output_vali = model(inputs_vali, tar_seq_vali)
    #     _, predicted_labels = torch.max(output_vali.cpu().data, 2)
    #
    #     sequece_label_a[i + 1, cur_slwin[10]:(cur_slwin[-1] + 1)] = predicted_labels[]
    #     sequece_label_b[i + 1, cur_slwin[10]:(cur_slwin[-1] + 1)] = predicted_labels[]
    #
    #     cm += confusion_matrix(labels_vali.numpy().reshape(-1, ), predicted_labels.view(-1, ),
    #                            labels=[0, 1, 2, 3, 4, 5, 6])
    #     correct_pred = (predicted_labels == labels_vali).sum().item()
    #     total_pred = predicted_labels.size(0) * predicted_labels.size(1)
    #     running_accuracy += correct_pred / total_pred
    #     torch.cuda.empty_cache()
    # average_accuracy = running_accuracy / len(sacro_test) * 100
    # print(average_accuracy)
    # print(cm / np.sum(cm, axis=1)[:, None] * 100)

if __name__ == '__main__':
    main()
