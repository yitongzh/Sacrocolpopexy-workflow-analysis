from seq2seq_LSTM import seq2seq_LSTM, seq2seq_LSTM_merge
from transformer.transformer import Transformer
import numpy as np

import pickle
import time
import psutil
import os
import scipy.io as scio
from tqdm import tqdm
import visdom
import random
import json

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.utils import data

from sklearn.metrics import confusion_matrix
from torchsummary import summary


class data_loder(data.Dataset):
    def __init__(self, train_video_list, seq_len=100, step=10):
        self.step = step
        self.seq_len = seq_len

        current_path = os.path.abspath(os.getcwd())

        self.train_video_list = train_video_list
        # self.validation_video_list = ['video' + str(i).zfill(2) for i in range(41, 61)]

        self.inputs_train = [self.video_loder(video_name) for video_name in self.train_video_list]
        # self.inputs_validation = [self.video_loder(video_name) for video_name in self.train_video_list]

        self.sliwins = [self.sliwin_idx_generator(len(dict['pred']), self.seq_len, self.step)
                        for dict in self.inputs_train]
        self.max_len = max([len(sliwin) for sliwin in self.sliwins])
        # self.sliwins_validation = [self.sliwin_idx_generator(len(dict['pred']), self.seq_len, self.step)
        #                            for dict in self.inputs_validation]

    def __getitem__(self, idx):
        input_dict = self.sliwin_idx_decoder(self.sliwins, self.inputs_train, idx)
        return input_dict

    def __len__(self):
        return self.max_len

    def sliwin_idx_generator(self, video_len, seq_len, step):
        # make a list that contains all the index for all sequences
        index = list(range(video_len))
        slwin_size = seq_len
        start_index = 0
        slinwin_list = []
        while start_index + slwin_size <= len(index):
            cur_slwin = index[start_index: start_index + slwin_size]
            slinwin_list.append(cur_slwin)
            start_index += step
        if start_index + slwin_size - step != len(index):
            cur_slwin = index[-slwin_size:]
            slinwin_list.append(cur_slwin)
        return slinwin_list

    def video_loder(self, video_name):
        open_path = os.path.join('/home/yitong/venv_yitong/cholec80_phase/data/whole', video_name)

        file = open(os.path.join(open_path, 'seq_pred.pickle'), 'rb')
        seq_pre = pickle.load(file)
        file.close()

        file = open(os.path.join(open_path, 'seq_true.pickle'), 'rb')
        seq_true = pickle.load(file)
        file.close()

        file = open(os.path.join(open_path, 'fc_list.pickle'), 'rb')
        fc_list = pickle.load(file)
        fc_list = [fc.unsqueeze(0).cpu() for fc in fc_list]
        file.close()
        input = {'true': seq_true,
                 'pred': seq_pre,
                 'fc_list': fc_list
                 }
        return input

    def sliwin_idx_decoder(self, sliwins, inputs, cur_idx):
        sequence_input_fc = torch.zeros(len(inputs), self.seq_len, 1200)
        sequence_true = torch.zeros(len(inputs), self.seq_len)
        sequence_input_pred = torch.zeros(len(inputs), self.seq_len)
        sequence_clip_num = torch.zeros(len(inputs), self.seq_len)

        for i, dict in enumerate(inputs):
            cur_idx_mod = cur_idx % len(sliwins[i])
            cur_sliwin = sliwins[i][cur_idx_mod]
            sequence_true[i, :] = torch.tensor(dict['true'][cur_sliwin[0]:cur_sliwin[-1] + 1])
            sequence_input_pred[i, :] = torch.tensor(dict['pred'][cur_sliwin[0]:cur_sliwin[-1] + 1])
            sequence_clip_num[i, :] = torch.tensor(cur_sliwin)
            for j, idx in enumerate(cur_sliwin):
                sequence_input_fc[i, j, :] = dict['fc_list'][idx]
        return {'fc': sequence_input_fc, 'true': sequence_true, 'pred': sequence_input_pred, 'clip_num': sequence_clip_num}


def sequence_loss(output, labels, device):
    # alpha = np.array([28.33, 11.23, 5.00, 2.09, 36.36, 11.01, 5.98]) / 100
    # loss_func = FocalLoss(7, alpha=alpha).to(device)
    # w = [0.1, 1.6, 0.5, 0.8, 2.0, 1, 1]
    # w = torch.tensor([0.1, 1.5, 0.3, 0.6, 2.0, 1, 1])
    w = torch.tensor([1.0, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
    # w = torch.tensor([20.0, 2, 11, 3, 20, 11, 24])

    loss_func = nn.NLLLoss(weight=w).to(device)
    l = output.size()[1]
    pred_labels = [torch.max(output[:, i, :].data, 1)[1] for i in range(l - 1)]
    pred_labels.insert(0, torch.max(output[:, 0, :].data, 1)[1])
    loss_cont = sum([loss_func(output[:, i, :], pred_labels[i]) for i in range(l)]) / l
    loss = sum([loss_func(output[:, i, :], labels[:, i]) for i in range(l)]) / l
    return loss + 0 * loss_cont


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


def eval(eval_list):
    model.eval()
    cm_eval = np.zeros((7, 7))
    running_loss = 0.0
    running_accuracy = 0.0
    valid_num = 0
    for video in tqdm(eval_list, ncols=80):
        for i in range(len(video)):
            inputs_dict_eval = video[i]
            inputs_eval = inputs_dict_eval['fc'].to(device)
            labels_eval = inputs_dict_eval['true'][:, 0:100].long().to(device)
            trg_seq_eval = inputs_dict_eval['true'][:, 0:100].long()
            # clip_num_seq_eval = inputs_dict_eval['clip_num'][:, 0:90].long()
            # clip_num_seq_eval = random_replace(clip_num_seq_eval, 0).long().to(device)
            trg_seq_eval = random_replace(trg_seq_eval, 0.3).long().to(device)
            with torch.no_grad():
                output_eval = model.forward(inputs_eval, trg_seq_eval)
                # output_eval = model.forward(inputs_eval, trg_seq_eval, clip_num_seq_eval)
            loss_eval = sequence_loss(output_eval, labels_eval, device)
            _, predicted_labels_eval = torch.max(output_eval.cpu().data, 2)
            cm_eval += confusion_matrix(labels_eval.cpu().numpy().reshape(-1, ), predicted_labels_eval.view(-1, ),
                                        labels=[0, 1, 2, 3, 4, 5, 6])
            correct_pred_eval = (predicted_labels_eval == labels_eval.cpu()).sum().item()
            total_pred_eval = predicted_labels_eval.size(0) * predicted_labels_eval.size(1)
            accuracy_eval = correct_pred_eval / total_pred_eval * 100
            running_accuracy += accuracy_eval
            running_loss += loss_eval
            valid_num += 1
    return torch.Tensor([running_loss / valid_num]), \
           torch.Tensor([running_accuracy / valid_num]), cm_eval / np.sum(cm_eval, axis=1)[:, None]


model_type = 'LSTM'
# model_type = 'transformer'

if model_type == 'LSTM':
    vis = visdom.Visdom(env='LSTM')
    print('Initializing the model')
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = seq2seq_LSTM(hidden_dim=1200, num_layers=3, tar_seq_dim=7).to(device)
    # model = seq2seq_LSTM_merge(hidden_dim=1200, num_layers=3).to(device)
    # model.load_state_dict(torch.load('./params/params_LSTM_seq2seq_half.pkl'))
    optimizer = optim.Adam(model.parameters(), lr=1e-6)  # , weight_decay=3e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
else:
    vis = visdom.Visdom(env='Transformer')
    print('Initializing the model')
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = Transformer(trg_pad_idx=6, n_trg_vocab=7,
                        d_word_vec=1200, d_model=1200, d_inner=1000,
                        n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=100).to(device)
    # model.load_state_dict(torch.load('./params/params_trans_save_point.pkl'))
    optimizer = optim.Adam(model.parameters(), lr=1e-6)  # 1e-5 for 75
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    # summary(model, [(10, 100, 1200), (10, 100)])

train_video_list = ['video' + str(i).zfill(2) for i in range(1, 41)]
validation_video_list = ['video' + str(i).zfill(2) for i in range(41, 61)]
data_sequence = data_loder(train_video_list)
validation_sequence_single_ls = [data_loder([video_name]) for video_name in validation_video_list]

accuracy_stas = []
loss_stas = []
counter = 0
best_accuracy = 0
folder_idx = 0
rebuild_slot = 1
evaluation_slot = 25
print('Start training')
start = time.time()
for epoch in range(500):
    # reconstruct the train set for every 7 epochs
    if epoch % rebuild_slot == 0 and epoch != 0:
        random.shuffle(train_video_list)
        data_sequence = data_loder(train_video_list)

    print('epoch:', epoch + 1)
    for i in tqdm(range(len(data_sequence)), ncols=80):
        counter += 1
        model.train()
        inputs_dict = data_sequence[i]
        inputs = inputs_dict['fc'].to(device)
        # labels = inputs_dict['true'][:, 0:100].long().to(device)
        # trg_seq = inputs_dict['true'][:, 0:100].long()

        labels = inputs_dict['true'][:, 0:100].long().to(device)
        trg_seq = inputs_dict['true'][:, 0:100].long()

        # labels = inputs_dict['true'][:, 0:100].long().to(device)
        # trg_seq = inputs_dict['pred'][:, 0:100].long()

        # labels = inputs_dict['true'][:, 10:100].long().to(device)
        # trg_seq = inputs_dict['pred'][:, 0:90].long()
        # clip_num_seq = inputs_dict['clip_num'][:, 0:90].long()
        # clip_num_seq = random_replace(clip_num_seq, 0).long().to(device)
        # introduce noise into the target sequence
        trg_seq = random_replace(trg_seq, 0.3).long().to(device)  # 0 for no noise

        optimizer.zero_grad()
        # trg_seq = (torch.ones(10, 300) * 6).long().to(device)
        output = model.forward(inputs, trg_seq)
        # output = model.forward(inputs, trg_seq, clip_num_seq)
        train_loss = sequence_loss(output, labels, device)
        train_loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)
        optimizer.step()

    # if counter % evaluation_slot == 0:
    train_sequence_single_ls = [data_loder([video_name]) for video_name in train_video_list]
    y_train_loss, y_train_acc, _ = eval(train_sequence_single_ls)
    y_batch_loss, y_batch_acc, cm = eval(validation_sequence_single_ls)

    # visualization in visdom
    x = torch.Tensor([counter])
    txt1 = ''.join(['t%d:%d ' % (i, np.sum(cm, axis=1)[i]) for i in range(len(np.sum(cm, axis=1)))])
    txt2 = ''.join(['p%d:%d ' % (i, np.sum(cm, axis=0)[i]) for i in range(len(np.sum(cm, axis=0)))])
    vis.text((txt1 + '<br>' + txt2), win='summary', opts=dict(title='Summary'))
    anot = np.sum(np.diag(cm)) / 7 * 100
    vis.heatmap(X=cm, win='heatmap', opts=dict(title='confusion matrix',
                                               rownames=['t1', 't2', 't3', 't4', 't5', 't6', 't7'],
                                               columnnames=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']))
    vis.line(X=x, Y=np.column_stack((y_train_acc, y_batch_acc)), win='accuracy', update='append',
             opts=dict(title='accuracy', showlegend=True, legend=['train', 'valid']))
    vis.line(X=x, Y=np.column_stack((y_train_loss, y_batch_loss)), win='loss', update='append',
             opts=dict(title='loss', showlegend=True, legend=['train', 'valid']))

    # Save point
    if anot > best_accuracy:
        best_accuracy = anot
        if model_type == 'LSTM':
            torch.save(model.state_dict(), './params/params_LSTM_save_point.pkl')
        else:
            torch.save(model.state_dict(), './params/params_trans_save_point.pkl')
        print('the current best accuracy without transition phase is: %.3f %%' % anot)
        txt = 'the current best accuracy is: %.3f%%, with' % best_accuracy
        x = list(np.diag(cm) * 100)
        txt3 = ''.join(['Phase %d accuracy:%d%%   ' % (i + 1, x[i]) for i in range(len(x))])
        vis.text((txt + '<br>' + txt3), win='cur_best', opts=dict(title='Current Best'))
        vis.heatmap(X=cm, win='best_cm', opts=dict(title='Best confusion matrix',
                                                   rownames=['t1', 't2', 't3', 't4', 't5', 't6', 't7'],
                                                   columnnames=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']))
    # viz.image(img, opts=dict(title='Example input'))

    # print('[Final results of epoch: %d] '
    #       ' train loss: %.3f '
    #       ' train accuracy: %.3f %%'
    #       ' validation loss: %.3f '
    #       ' validation accuracy: %.3f %%'
    #       ' The current learning rate is: %f'
    #       % (epoch + 1, train_loss, train_accuracy, batch_loss, batch_accuracy, optimizer.param_groups[0]['lr']))
    scheduler.step()
elapsed = (time.time() - start)
print("Training finished, time used:", elapsed / 60, 'min')

# torch.save(model.state_dict(), 'params_endo3d.pkl')

data_path = 'results_output.mat'
scio.savemat(data_path, {'accuracy': np.asarray(accuracy_stas), 'loss': np.asarray(loss_stas)})
