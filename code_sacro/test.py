#! /usr/bin/env python
from model import Endo3D_1vo, Endo3D_1vo1, Endo3D_1vo2
from seq2seq_LSTM import seq2seq_LSTM
from transformer.transformer import Transformer
from utils.sacro_loder import sacro_loder
import numpy as np
import visdom
from tqdm import tqdm

import time
import os
import scipy.io as scio
import pickle

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
from torch.utils import data

from sklearn.metrics import confusion_matrix


class data_loder(data.Dataset):
    def __init__(self, mode='train', datatype='current', train_epoch_size=1000, validation_epoch_size=600,
                 building_block=True, sliwin_sz=300):
        self.batch_num = train_epoch_size
        self.validation_batch_size = validation_epoch_size
        self.mode = mode
        self.datatype = datatype
        self.folder = 'folder1'

        self.validation_folder = 'folder1'
        self.div = 'div1'

        if building_block:
            # start = time.time()
            self.build_epoch()
            self.build_validation()
            # elapsed = (time.time() - start)
            # print("Data loded, time used:", elapsed)

    def __getitem__(self, idx):
        if self.mode == 'train':
            labels = {'past': self.epoch_train_labels_past[idx], 'current': self.epoch_train_labels_cur[idx],
                      'future': self.epoch_train_labels_future[idx]}
            inputs = self.epoch_train_inputs[idx]
        elif self.mode == 'validation':
            labels = {'past': self.epoch_validation_labels_past[idx], 'current': self.epoch_validation_labels_cur[idx],
                      'future': self.epoch_validation_labels_future[idx]}
            inputs = self.epoch_validation_inputs[idx]
        return labels, inputs

    def __len__(self):
        if self.mode == 'train':
            return self.batch_num
        elif self.mode == 'validation':
            return self.validation_batch_size

    def build_epoch(self):
        print('building the training set...')
        self.mode = 'train'

        train_input = open('./data/sacro_sequence/train/' + self.div + '/' + self.folder + '/train_input.pickle', 'rb')
        self.epoch_train_inputs = pickle.load(train_input)
        train_input.close()

        label_past = open('./data/sacro_sequence/train/' + self.div + '/' + self.folder + '/label_past.pickle', 'rb')
        epoch_train_labels_past = pickle.load(label_past)
        self.epoch_train_labels_past = [torch.FloatTensor(item) for item in epoch_train_labels_past]
        label_past.close()

        label_cur = open('./data/sacro_sequence/train/' + self.div + '/' + self.folder + '/label_curr.pickle', 'rb')
        epoch_train_labels_cur = pickle.load(label_cur)
        self.epoch_train_labels_cur = [torch.FloatTensor(item) for item in epoch_train_labels_cur]
        label_cur.close()

        label_future = open('./data/sacro_sequence/train/' + self.div + '/' + self.folder + '/label_future.pickle',
                            'rb')
        epoch_train_labels_future = pickle.load(label_future)
        self.epoch_train_labels_future = [torch.FloatTensor(item) for item in epoch_train_labels_future]
        label_future.close()

    def build_validation(self):
        print('building the validation set...')
        self.mode = 'validation'
        folder = self.validation_folder

        train_input = open('./data/sacro_sequence/validation/' + folder + '/validation_input.pickle', 'rb')
        self.epoch_validation_inputs = pickle.load(train_input)
        train_input.close()

        label_past = open('./data/sacro_sequence/validation/' + folder + '/label_past.pickle', 'rb')
        epoch_validation_labels_past = pickle.load(label_past)
        self.epoch_validation_labels_past = [torch.FloatTensor(item)
                                             for item in epoch_validation_labels_past]
        label_past.close()

        label_curr = open('./data/sacro_sequence/validation/' + folder + '/label_curr.pickle', 'rb')
        epoch_validation_labels_cur = pickle.load(label_curr)
        self.epoch_validation_labels_cur = [torch.FloatTensor(item)
                                            for item in epoch_validation_labels_cur]
        label_curr.close()

        label_future = open('./data/sacro_sequence/validation/' + folder + '/label_future.pickle', 'rb')
        epoch_validation_labels_future = pickle.load(label_future)
        self.epoch_validation_labels_future = [torch.FloatTensor(item)
                                               for item in epoch_validation_labels_future]
        label_future.close()


def sequence_loss(output, labels, device):
    # alpha = np.array([28.33, 11.23, 5.00, 2.09, 36.36, 11.01, 5.98]) / 100
    # loss_func = FocalLoss(7, alpha=alpha).to(device)
    # w = [0.1, 1.6, 0.5, 0.8, 2.0, 1, 1]
    # w = torch.tensor([0.1, 2.4, 0.3, 0.6, 2.4, 1, 1])
    w = torch.tensor([0.1, 1, 1, 1, 1, 1, 1])
    loss_func = nn.NLLLoss(weight=w).to(device)
    l = output.size()[1]
    pred_labels = [torch.max(output[:, i, :].data, 1)[1] for i in range(l - 1)]
    pred_labels.insert(0, torch.max(output[:, 0, :].data, 1)[1])
    loss_cont = sum([loss_func(output[:, i, :], pred_labels[i]) for i in range(l)]) / l
    loss = sum([loss_func(output[:, i, :], labels[:, i]) for i in range(l)]) / l
    return loss + 0.5 * loss_cont


def main():
    vis = visdom.Visdom(env='test')
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    sequentiaol_model_name = 'transformer'
    sequentiaol_model_name = 'LSTM'
    folder = 'div1'
    print('The sequential model chosen is: %s' % sequentiaol_model_name)
    if sequentiaol_model_name == 'transformer':
        model = Transformer(trg_pad_idx=6, n_trg_vocab=7,
                            d_word_vec=1200, d_model=1200, d_inner=1000,
                            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=100).to(device)
        model.load_state_dict(torch.load('./params/cross_validation/' + folder + '/params_trans_75.pkl'))
        # model.load_state_dict(torch.load('./params/params_trans_save_point.pkl'))
        slin_sz = 30
    else:
        model = seq2seq_LSTM(hidden_dim=1200, num_layers=3).to(device)
        model.load_state_dict(torch.load('./params/cross_validation/' + folder + '/params_LSTM_90.pkl'))
        # model.load_state_dict(torch.load('./params/params_LSTM_save_point.pkl'))
        slin_sz = 17

    # loading the training and validation set
    print('loading data')
    start = time.time()
    # sacro = sacro_loder(batch_num=10, validation_batch_size=500)
    sacro = data_loder(train_epoch_size=1000, validation_epoch_size=1000)
    train_loader = data.DataLoader(sacro, 50, shuffle=True)
    valid_loader = data.DataLoader(sacro, 50, shuffle=True)
    elapsed = (time.time() - start)
    print("Data loded, time used:", elapsed)

    # Initializing necessary components
    loss_func = nn.NLLLoss().to(device)

    # Evaluation on validation set
    model.eval()
    sacro.mode = 'validation'
    running_loss = 0.0
    running_accuracy = 0.0
    valid_num = 0
    cm = np.zeros((7, 7))
    for labels_val_, inputs_val in valid_loader:
        inputs_val = inputs_val.to(device)
        # labels_val = labels_val_['current'][:, 25:100].long().to(device)
        # trg_seq_val = labels_val_['current'][:, 0:75].long().to(device)
        # labels_val = labels_val_['current'][:, 0:100].long().to(device)
        # trg_seq_val = labels_val_['current'][:, 0:100].long().to(device)
        labels_val = labels_val_['current'][:, 10:100].long().to(device)
        trg_seq_val = labels_val_['current'][:, 0:90].long().to(device)
        # trg_seq_val = labels_val_[:, 0:25].long().to(device)
        # trg_seq_val = (torch.ones(10, 300) * 6).long().to(device)
        # introduce noise into the target sequence

        size = int(trg_seq_val.size(0) * trg_seq_val.size(1) * 0.3)
        a = np.random.randint(trg_seq_val.size(0), size=size)
        b = np.random.randint(trg_seq_val.size(1), size=size)
        trg_seq_val[a, b] = torch.from_numpy(np.random.randint(7, size=size)).long().to(device)


        output = model.forward(inputs_val, trg_seq_val)
        valid_loss = sequence_loss(output, labels_val, device)

        # Calculate the loss with new parameters
        running_loss += valid_loss.item()
        # current_loss = running_loss / (batch_counter + 1)
        _, predicted_labels = torch.max(output.cpu().data, 2)
        cm += confusion_matrix(labels_val.cpu().numpy().reshape(-1, ), predicted_labels.view(-1, ),
                               labels=[0, 1, 2, 3, 4, 5, 6])
        correct_pred = (predicted_labels == labels_val.cpu()).sum().item()
        total_pred = predicted_labels.size(0) * predicted_labels.size(1)
        accuracy = correct_pred / total_pred
        running_accuracy += accuracy
        valid_num += 1
        # current_accuracy = running_accuracy / (batch_counter + 1) * 100
    batch_loss = running_loss / valid_num
    batch_accuracy = running_accuracy / valid_num * 100

    # visualization in visdom
    txt1 = ''.join(['t%d:%d ' % (i, np.sum(cm, axis=1)[i]) for i in range(len(np.sum(cm, axis=1)))])
    txt2 = ''.join(['p%d:%d ' % (i, np.sum(cm, axis=0)[i]) for i in range(len(np.sum(cm, axis=0)))])
    vis.text((txt1 + '<br>' + txt2), win='summary', opts=dict(title='Summary'))
    # cm = cm / np.sum(cm, axis=1)
    cm_inner = cm[0:6, 0:6]
    cm_inner = cm_inner / np.sum(cm_inner, axis=1)
    x = list(np.diag(cm_inner) * 100)
    txt3 = ''.join(['Phase %d accuracy:%d%%   ' % (i + 1, x[i]) for i in range(len(x))])
    vis.text(txt3, win='cur_best', opts=dict(title='Current Best'))
    anot = np.sum(np.diag(cm)[1:6]) / 5 * 100
    vis.heatmap(X=cm_inner, win='heatmap', opts=dict(title='confusion matrix',
                                               rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
                                               columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))



if __name__ == '__main__':
    main()
