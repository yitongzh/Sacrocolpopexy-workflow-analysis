#! /usr/bin/env python
from model import Endo3D_for_sequence
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
import pickle

import torch.nn as nn
from torch.utils import data

from sklearn.metrics import confusion_matrix


class data_loder(data.Dataset):
    def __init__(self, mode='validation', datatype='current', train_epoch_size=1000, validation_epoch_size=600,
                 building_block=True, sliwin_sz=300):
        self.batch_num = train_epoch_size
        self.validation_batch_size = validation_epoch_size
        self.mode = mode
        self.datatype = datatype
        self.folder = 'folder1'

        if building_block:
            # start = time.time()
            # self.build_epoch()
            self.build_validation()
            # elapsed = (time.time() - start)
            # print("Data loded, time used:", elapsed)

    def __getitem__(self, idx):
        if self.mode == 'train':
            if self.datatype == 'past':
                labels = self.epoch_train_labels_past[idx]
            elif self.datatype == 'current':
                labels = self.epoch_train_labels_cur[idx]
                tar_sequence = self.epoch_train_labels_past[idx]
            elif self.datatype == 'future':
                labels = self.epoch_train_labels_future[idx]
            inputs = self.epoch_train_inputs[idx]
        elif self.mode == 'validation':
            if self.datatype == 'past':
                labels = self.epoch_validation_labels_past[idx]
            elif self.datatype == 'current':
                labels = self.epoch_validation_labels_cur[idx]
                tar_sequence = self.epoch_validation_labels_past[idx]
            elif self.datatype == 'future':
                labels = self.epoch_validation_labels_future[idx]
            inputs = self.epoch_validation_inputs[idx]
        return labels, inputs, tar_sequence

    def __len__(self):
        if self.mode == 'train':
            return self.batch_num
        elif self.mode == 'validation':
            return self.validation_batch_size

    def build_epoch(self):
        print('building the training set...')
        self.mode = 'train'

        train_input = open('./data/sacro_sequence/train/' + self.folder + '/train_input.pickle', 'rb')
        self.epoch_train_inputs = pickle.load(train_input)
        train_input.close()

        label_past = open('./data/sacro_sequence/train/' + self.folder + '/label_past.pickle', 'rb')
        epoch_train_labels_past = pickle.load(label_past)
        self.epoch_train_labels_past = [torch.FloatTensor(item) for item in epoch_train_labels_past]
        label_past.close()

        label_cur = open('./data/sacro_sequence/train/' + self.folder + '/label_curr.pickle', 'rb')
        epoch_train_labels_cur = pickle.load(label_cur)
        self.epoch_train_labels_cur = [torch.FloatTensor(item) for item in epoch_train_labels_cur]
        label_cur.close()

        label_future = open('./data/sacro_sequence/train/' + self.folder + '/label_future.pickle', 'rb')
        epoch_train_labels_future = pickle.load(label_future)
        self.epoch_train_labels_future = [torch.FloatTensor(item) for item in epoch_train_labels_future]
        label_future.close()

    def build_validation(self):
        print('building the validation set...')
        self.mode = 'validation'
        folder = 'folder1'

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

        # train_input = open('./temp/whole_input.pickle', 'rb')
        # self.epoch_validation_inputs = pickle.load(train_input)
        # train_input.close()
        #
        # label_curr = open('./temp/whole_labels.pickle' + folder + '/label_curr.pickle', 'rb')
        # epoch_validation_labels_cur = pickle.load(label_curr)
        # self.epoch_validation_labels_cur = [torch.FloatTensor(item)
        #                                     for item in epoch_validation_labels_cur]
        # label_curr.close()

        # validation_input = open('./temp/whole_input.pickle', 'wb')
        # whole_inputs = pickle.load(validation_input)
        # validation_input.close()
        # validation_labels = open('./temp/whole_labels.pickle', 'wb')
        # whole_labels = pickle.load(validation_labels)
        # validation_input.close()


def sequence_loss(output, labels, device):
    loss_func = nn.NLLLoss().to(device)
    l = output.size()[1]
    loss = sum([loss_func(output[:, i, :], labels[:, i]) for i in range(l)]) / l
    return loss


def main():
    vis = visdom.Visdom(env='test_transformer')
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    print('Initializing the model')
    start = time.time()
    Endo3D_model = Endo3D_for_sequence().to(device)
    Endo3D_model.load_state_dict(torch.load('./params/params_endo3d_for_sequence.pkl'))
    Seq_model = Transformer(trg_pad_idx=6, n_trg_vocab=7,
                        d_word_vec=4200, d_model=4200, d_inner=3000,
                        n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=300).to(device)
    Seq_model.load_state_dict(torch.load('./params/params_trans_cc.pkl'))
    Endo3D_model.eval()
    Seq_model.eval()
    elapsed = (time.time() - start)
    print("Model initialized, time used:", elapsed)

    # loading the training and validation set
    print('loading data')
    start = time.time()
    # sacro = sacro_loder(batch_num=10, validation_batch_size=500)
    sacro = sequence_loder()
    # video_name = '26d9e19b-59ed-4bbb-906a-9953b5d2c825'
    # video_name = 'a3347182-4364-43f8-8992-2409b1b66d31'
    # video_name = 'a90983a1-2329-4019-96e8-949493c3fc24'
    # video_name = 'e6a9f8f7-8024-42d5-aad5-ef726d353f3b'
    # sacro.whole_len_output(video_name)
    sacro = data_loder(train_epoch_size=1000, validation_epoch_size=600)
    whole_loder = data.DataLoader(sacro, 10, shuffle=True)

    elapsed = (time.time() - start)
    print("Data loded, time used:", elapsed)
    # Initializing necessary components
    loss_func = nn.NLLLoss().to(device)

    # Evaluation on validation set
    running_loss = 0.0
    running_accuracy = 0.0
    valid_num = 0
    cm = np.zeros((6, 6))
    seq_pre = []
    seq_true = []
    for labels_val_, inputs_val, trg_seq_val in tqdm(whole_loder, ncols=80):
        inputs_val = inputs_val.to(device)
        labels_val = labels_val_.long().to(device)
        trg_seq_val = labels_val_.long().to(device)

        # introduce noise into the target sequence
        size = int(trg_seq_val.size(0) * trg_seq_val.size(1) * 0.3)
        a = np.random.randint(trg_seq_val.size(0), size=size)
        b = np.random.randint(trg_seq_val.size(1), size=size)
        trg_seq_val[a, b] = torch.from_numpy(np.random.randint(7, size=size)).long().to(device)

        # output_4200 = Endo3D_model.forward_cov(inputs_val)
        # print(output_4200.size())
        # tar_seq = (torch.ones(10, 300) * 6).long().to(device)
        output = Seq_model.forward(inputs_val, trg_seq_val)
        # _, predicted_labels = torch.max(output.data, 2)
        # output = Seq_model.forward(inputs_val, predicted_labels)

        valid_loss = sequence_loss(output, labels_val, device)

        # Calculate the loss with new parameters
        running_loss += valid_loss.item()
        # current_loss = running_loss / (batch_counter + 1)

        _, predicted_labels = torch.max(output.cpu().data, 2)
        cm += confusion_matrix(labels_val.cpu().numpy().reshape(-1, ), predicted_labels.view(-1, ),
                               labels=[0, 1, 2, 3, 4, 5])
        correct_pred = (predicted_labels == labels_val.cpu()).sum().item()
        for i in range(predicted_labels.numpy().shape[0]):
            seq_pre.append(predicted_labels.numpy()[i])
            seq_true.append(labels_val.cpu().numpy()[i])
        total_pred = predicted_labels.size(0) * predicted_labels.size(1)
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


if __name__ == '__main__':
    main()
