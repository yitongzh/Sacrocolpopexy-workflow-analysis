#! /usr/bin/env python
from seq2seq_LSTM import seq2seq_LSTM
import numpy as np

import pickle
import time
import psutil
import os
import scipy.io as scio
from tqdm import tqdm
import visdom
import random

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
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

        self.validation_folder = 'folder7'
        self.div = 'div7'

        if building_block:
            # start = time.time()
            self.build_epoch()
            self.build_validation()
            # elapsed = (time.time() - start)
            # print("Data loded, time used:", elapsed)

    def __getitem__(self, idx):
        if self.mode == 'train':
            labels = {'past': self.epoch_train_labels_past[idx],
                      'current': self.epoch_train_labels_cur[idx],
                      'future': self.epoch_train_labels_future[idx],
                      'pred': self.epoch_train_labels_pred[idx]}
            inputs = self.epoch_train_inputs[idx]
        elif self.mode == 'validation':
            labels = {'past': self.epoch_validation_labels_past[idx],
                      'current': self.epoch_validation_labels_cur[idx],
                      'future': self.epoch_validation_labels_future[idx],
                      'pred': self.epoch_validation_labels_pred[idx]}
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

        label_pred = open('./data/sacro_sequence/train/' + self.div + '/' + self.folder + '/label_pred.pickle',
                            'rb')
        epoch_train_labels_pred = pickle.load(label_pred)
        self.epoch_train_labels_pred = [torch.FloatTensor(item) for item in epoch_train_labels_pred]
        label_pred.close()

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

        label_pred = open('./data/sacro_sequence/validation/' + folder + '/label_pred.pickle',
                            'rb')
        epoch_validation_labels_pred = pickle.load(label_pred)
        self.epoch_validation_labels_pred = [torch.FloatTensor(item) for item in epoch_validation_labels_pred]
        label_pred.close()


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


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
        del(a[idx])
        del(b[idx])
    trg_seq[a, b] = torch.from_numpy(np.random.randint(7, size=size)).long()
    return trg_seq


# def sequence_loss(output, labels, device):
#     # alpha = np.array([28.33, 11.23, 5.00, 2.09, 36.36, 11.01, 5.98]) / 100
#     # loss_func = FocalLoss(7, alpha=alpha).to(device)
#     loss_func = nn.NLLLoss().to(device)
#     l = output.size()[1]
#     loss = sum([loss_func(output[:, i, :], labels[:, i]) for i in range(l)]) / l
#     return loss

def sequence_loss(output, labels, device):
    # alpha = np.array([28.33, 11.23, 5.00, 2.09, 36.36, 11.01, 5.98]) / 100
    # loss_func = FocalLoss(7, alpha=alpha).to(device)
    # w = [0.1, 1.6, 0.5, 0.8, 2.0, 1, 1]
    # w = torch.tensor([0.1, 1.5, 0.3, 0.6, 2.0, 1, 1])
    w = torch.tensor([0.1, 1, 1, 1, 1, 1, 1])
    loss_func = nn.NLLLoss(weight=w).to(device)
    l = output.size()[1]
    pred_labels = [torch.max(output[:, i, :].data, 1)[1] for i in range(l - 1)]
    pred_labels.insert(0, torch.max(output[:, 0, :].data, 1)[1])
    loss_cont = sum([loss_func(output[:, i, :], pred_labels[i]) for i in range(l)]) / l
    loss = sum([loss_func(output[:, i, :], labels[:, i]) for i in range(l)]) / l
    return loss + 0 * loss_cont


def main():
    vis = visdom.Visdom(env='LSTM')
    print('Initializing the model')
    start = time.time()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = seq2seq_LSTM(hidden_dim=1200, num_layers=3).to(device)
    elapsed = (time.time() - start)
    # model.load_state_dict(torch.load('./params/params_LSTM_3layer.pkl'))
    print("Model initialized, time used:", elapsed)

    # load pre-trained parameters
    # Endo3D_state_dict = model.state_dict()
    # pre_state_dict = torch.load('./params/params_endo3d_1vo.pkl')
    # new_state_dict = {k: v for k, v in pre_state_dict.items() if k in Endo3D_state_dict}
    # Endo3D_state_dict.update(new_state_dict)
    # model.load_state_dict(Endo3D_state_dict)
    # model.load_state_dict(torch.load('./params/params_endo3d_1vo2.pkl'))

    # loading the training and validation set
    print('loading data')
    start = time.time()
    evaluation_slot = 25
    rebuild_slot = 1
    sacro = data_loder(train_epoch_size=1000, validation_epoch_size=1000)
    print('The current folder is :' + sacro.div)
    train_loader = data.DataLoader(sacro, 10, shuffle=True)
    valid_loader = data.DataLoader(sacro, 10, shuffle=True)
    folders = ['folder1', 'folder2', 'folder3', 'folder4', 'folder5']
    # folders = ['folder6', 'folder7', 'folder8', 'folder9', 'folder10']
    elapsed = (time.time() - start)
    print("Data loded, time used:", elapsed)

    # Initializing necessary components
    # loss_func = nn.NLLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # , weight_decay=3e-5)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

    # optimizer = optim.Adam(model.parameters(), lr=5e-6)  # , weight_decay=3e-5)
    # # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    accuracy_stas = []
    loss_stas = []
    counter = 0
    best_accuracy = 0
    folder_idx = 0

    print('Start training')
    start = time.time()
    for epoch in range(100):
        # reconstruct the train set for every 7 epochs
        if epoch % rebuild_slot == 0 and epoch != 0:
            folder_idx += 1
            if folder_idx == 5:
                folder_idx = 0
            sacro.folder = folders[folder_idx]
            print('The current training folder is: %s' % folders[folder_idx])
            sacro.build_epoch()
            # sacro.build_validation()

        print('epoch:', epoch + 1)
        sacro.mode = 'train'
        for labels_train, inputs in tqdm(train_loader, ncols=80):
            counter += 1
            model.train()
            inputs = inputs.to(device)
            labels = labels_train['current'][:, 0:100].long().to(device)
            trg_seq = labels_train['current'][:, 0:100].long()

            # labels = labels_train['current'][:, 10:100].long().to(device)
            # trg_seq = labels_train['current'][:, 0:90].long()

            # labels = labels_train['current'][:, 0:100].long().to(device)
            # trg_seq = labels_train['pred'][:, 0:100].long()

            # labels = labels_train['current'][:, 10:100].long().to(device)
            # trg_seq = labels_train['pred'][:, 0:90].long()
            # introduce noise into the target sequence
            trg_seq = random_replace(trg_seq, 0.4).long().to(device) # 0 for no noise

            optimizer.zero_grad()
            # trg_seq = (torch.ones(10, 300) * 6).long().to(device)
            output = model.forward(inputs, trg_seq)
            train_loss = sequence_loss(output, labels, device)
            train_loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 10)
            optimizer.step()

            if counter % evaluation_slot == 0:
                # Evaluation on training set
                _, predicted_labels = torch.max(output.cpu().data, 2)
                correct_pred = (predicted_labels == labels.cpu()).sum().item()
                total_pred = predicted_labels.size(0) * predicted_labels.size(1)
                train_accuracy = correct_pred / total_pred * 100

                # visualization in visdom
                y_train_acc = torch.Tensor([train_accuracy])
                y_train_loss = torch.Tensor([train_loss.item()])

                # Evaluation on validation set
                model.eval()
                sacro.mode = 'validation'
                running_loss = 0.0
                running_accuracy = 0.0
                valid_num = 0
                cm = np.zeros((7, 7))
                for labels_val_, inputs_val in valid_loader:
                    inputs_val = inputs_val.to(device)
                    labels_val = labels_val_['current'][:, 0:100].long().to(device)
                    trg_seq_val = labels_val_['current'][:, 0:100].long()

                    # labels_val = labels_val_['current'][:, 10:100].long().to(device)
                    # trg_seq_val = labels_val_['current'][:, 0:90].long()

                    # labels_val = labels_val_['current'][:, 0:100].long().to(device)
                    # trg_seq_val = labels_val_['pred'][:, 0:100].long()

                    # labels_val = labels_val_['current'][:, 10:100].long().to(device)
                    # trg_seq_val = labels_val_['pred'][:, 0:90].long()

                    trg_seq_val = random_replace(trg_seq_val, 0.4).long().to(device)
                    # trg_seq_val = labels_val_[:, 0:25].long().to(device)
                    # trg_seq_val = (torch.ones(10, 300) * 6).long().to(device)
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
                sacro.mode = 'train'

                # save the loss and accuracy
                accuracy_stas.append(batch_accuracy)
                loss_stas.append(batch_loss)

                # visualization in visdom
                x = torch.Tensor([counter])
                y_batch_acc = torch.Tensor([batch_accuracy])
                y_batch_loss = torch.Tensor([batch_loss])
                txt1 = ''.join(['t%d:%d ' % (i, np.sum(cm, axis=1)[i]) for i in range(len(np.sum(cm, axis=1)))])
                txt2 = ''.join(['p%d:%d ' % (i, np.sum(cm, axis=0)[i]) for i in range(len(np.sum(cm, axis=0)))])
                vis.text((txt1 + '<br>' + txt2), win='summary', opts=dict(title='Summary'))
                cm = cm / np.sum(cm, axis=1)
                anot = np.sum(np.diag(cm)[1:6]) / 5 * 100
                vis.heatmap(X=cm, win='heatmap', opts=dict(title='confusion matrix',
                                                           rownames=['t0', 't1', 't2', 't3', 't4', 't5', 't_not'],
                                                           columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p_not']))
                vis.line(X=x, Y=np.column_stack((y_train_acc, y_batch_acc, anot)), win='accuracy', update='append',
                         opts=dict(title='accuracy', showlegend=True, legend=['train', 'valid', 'valid_no_T']))
                vis.line(X=x, Y=np.column_stack((y_train_loss, y_batch_loss)), win='loss', update='append',
                         opts=dict(title='loss', showlegend=True, legend=['train', 'valid']))

                # Save point
                if anot > best_accuracy:
                    best_accuracy = anot
                    torch.save(model.state_dict(), './params/params_LSTM_save_point.pkl')
                    print('the current best accuracy without transition phase is: %.3f %%' % anot)
                    txt = 'the current best accuracy is: %.3f%%, with' % best_accuracy
                    x = list(np.diag(cm)[1:6] * 100)
                    txt3 = ''.join(['Phase %d accuracy:%d%%   ' % (i + 1, x[i]) for i in range(len(x))])
                    vis.text((txt + '<br>' + txt3), win='cur_best', opts=dict(title='Current Best'))
                # viz.image(img, opts=dict(title='Example input'))

        print('[Final results of epoch: %d] '
              ' train loss: %.3f '
              ' train accuracy: %.3f %%'
              ' validation loss: %.3f '
              ' validation accuracy: %.3f %%'
              ' The current learning rate is: %f'
              % (epoch + 1, train_loss, train_accuracy, batch_loss, batch_accuracy, optimizer.param_groups[0]['lr']))
        scheduler.step()
    elapsed = (time.time() - start)
    print("Training finished, time used:", elapsed / 60, 'min')

    # torch.save(model.state_dict(), 'params_endo3d.pkl')

    data_path = 'results_output.mat'
    scio.savemat(data_path, {'accuracy': np.asarray(accuracy_stas), 'loss': np.asarray(loss_stas)})


if __name__ == '__main__':
    main()
