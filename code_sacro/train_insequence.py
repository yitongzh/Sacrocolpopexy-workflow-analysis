from seq2seq_LSTM import seq2seq_LSTM
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
import requests


class data_loder(data.Dataset):
    def __init__(self, seq_num=200, seq_len=100):
        self.seq_num = seq_num
        self.seq_len = seq_len
        self.mode = 'train'

        self.folder = folders[fold_num]
        self.KF_Folder = KF_Folders[fold_num]

        current_path = os.path.abspath(os.getcwd())
        videos_path = os.path.join(current_path, 'data/sacro_jpg')
        with open(os.path.join(videos_path, 'dataset_' + self.folder + '.json'), 'r') as json_data:
            temp = json.load(json_data)
        self.train_video_list = temp['train']
        self.validation_video_list = temp['validation']

        self.inputs_train = [self.video_loder(video_name) for video_name in self.train_video_list]
        self.inputs_validation = [self.video_loder(video_name) for video_name in self.train_video_list]

        self.sliwins_train = [self.sliwin_idx_generator(len(dict['pred']), self.seq_len, self.seq_num)
                              for dict in self.inputs_train]
        self.sliwins_validation = [self.sliwin_idx_generator(len(dict['pred']), self.seq_len, self.seq_num)
                                   for dict in self.inputs_validation]

    def __getitem__(self, idx):
        if self.mode == 'train':
            input_dict = self.sliwin_idx_decoder(self.sliwins_train, self.inputs_train, idx)
            return input_dict
        elif self.mode == 'validation':
            input_dict = self.sliwin_idx_decoder(self.sliwins_validation, self.inputs_validation, idx)
            return input_dict

    def __len__(self):
        return self.seq_num

    def sliwin_idx_generator(self, video_len, seq_len, seq_num):
        index = list(range(video_len))
        step_size = round(video_len / (seq_num + 1)) - 1

        start_index = 0
        slinwin_list = []
        while start_index + seq_len <= video_len:
            cur_slwin = index[start_index: start_index + self.seq_len]
            slinwin_list.append(cur_slwin)
            start_index += step_size

        while len(slinwin_list) != seq_num:
            idx = random.choice(range(len(slinwin_list) - 1))
            del slinwin_list[idx]

        untouched = slinwin_list[-1][-1] + 1 - video_len

        slinwin_list[-1] = [slinwin_list[-1][i] - int(untouched) for i in range(len(slinwin_list[-1]))]

        return slinwin_list

    def video_loder(self, video_name):
        open_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_sequence/whole',
                                 self.KF_Folder, video_name)

        file = open(os.path.join(open_path, 'seq_pred_c3d.pickle'), 'rb')
        seq_pre = pickle.load(file)
        file.close()

        file = open(os.path.join(open_path, 'seq_true.pickle'), 'rb')
        seq_true = pickle.load(file)
        file.close()

        file = open(os.path.join(open_path, 'fc_list.pickle'), 'rb')
        fc_list = pickle.load(file)
        # fc_list = [item.cpu() for item in fc_list]
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

        for i, dict in enumerate(inputs):
            cur_sliwin = sliwins[i][cur_idx]
            sequence_true[i, :] = torch.tensor(dict['true'][cur_sliwin[0]:cur_sliwin[-1] + 1])
            sequence_input_pred[i, :] = torch.tensor(dict['pred'][cur_sliwin[0]:cur_sliwin[-1] + 1])
            for j, idx in enumerate(cur_sliwin):
                sequence_input_fc[i, j, :] = dict['fc_list'][idx]
        return {'fc': sequence_input_fc, 'true': sequence_true, 'pred': sequence_input_pred}

    def build_epoch(self):
        self.sliwins_train = [self.sliwin_idx_generator(len(dict['pred']), self.seq_len, self.seq_num)
                              for dict in self.inputs_train]

    def build_validation(self):
        self.sliwins_validation = [self.sliwin_idx_generator(len(dict['pred']), self.seq_len, self.seq_num)
                                   for dict in self.inputs_validation]


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


class trans_batch_first(nn.Module):
    def __init__(self, nhead=8, num_encoder_layers=6, d_model=1200, dim_feedforward=1000):
        super(trans_batch_first, self).__init__()
        self.model1 = nn.Transformer(nhead=nhead, num_encoder_layers=num_encoder_layers,
                                     d_model=d_model, dim_feedforward=dim_feedforward)

    def forward(self, x, tar_seq):
        print(x.size())
        print(tar_seq.size())
        output_ = model(x.permute(1, 0, 2), tar_seq.permute(1, 0, 2))
        return output_


def send_notice(event_name, text, key='d_kRy2XrftABFT4w0v7TeW'):
    report = {"value1": text}
    requests.post(f'https://maker.ifttt.com/trigger/{event_name}/with/key/{key}', data=report)


def main():
    if model_type == 'LSTM':
        vis = visdom.Visdom(env='LSTM')
        print('Initializing the model')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = seq2seq_LSTM(hidden_dim=1200, num_layers=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)  # , weight_decay=3e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
        max_epoch = 50
    else:
        vis = visdom.Visdom(env='Transformer')
        print('Initializing the model')
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        model = Transformer(trg_pad_idx=6, n_trg_vocab=7,
                            d_word_vec=1200, d_model=1200, d_inner=1000,
                            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=100).to(device)
        # model = trans_batch_first(nhead=8, num_encoder_layers=6, d_model=1200, dim_feedforward=1000).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)  # 1e-5 for 75
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
        # summary(model, [(10, 100, 1200), (10, 100)])
        max_epoch = 100

    data_sequence = data_loder()
    print('The current folder is :' + data_sequence.folder)
    print('The current model is :' + model_type)

    accuracy_stas = []
    loss_stas = []
    counter = 0
    best_accuracy = 0
    folder_idx = 0
    rebuild_slot = 1
    evaluation_slot = 25
    print('Start training')
    start = time.time()
    for epoch in range(max_epoch):
        # reconstruct the train set for every 7 epochs
        if epoch % rebuild_slot == 0 and epoch != 0:
            folder_idx += 1
            if folder_idx == 5:
                folder_idx = 0
            data_sequence.build_epoch()
            data_sequence.build_validation()

        print('epoch:', epoch + 1)
        data_sequence.mode = 'train'
        for inputs_dict in tqdm(data_sequence, ncols=80):
            counter += 1
            model.train()
            inputs = inputs_dict['fc'].to(device)
            # labels = inputs_dict['true'][:, 0:100].long().to(device)
            # trg_seq = inputs_dict['true'][:, 0:100].long()

            labels = inputs_dict['true'][:, 10:100].long().to(device)
            trg_seq = inputs_dict['true'][:, 0:90].long()

            # labels = inputs_dict['true'][:, 0:100].long().to(device)
            # trg_seq = inputs_dict['pred'][:, 0:100].long()

            # labels = inputs_dict['true'][:, 10:100].long().to(device)
            # trg_seq = inputs_dict['pred'][:, 0:90].long()
            # introduce noise into the target sequence
            trg_seq = random_replace(trg_seq, 0).long().to(device)  # 0 for no noise

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
                data_sequence.mode = 'validation'
                running_loss = 0.0
                running_accuracy = 0.0
                valid_num = 0
                cm = np.zeros((7, 7))
                for validation_dict in data_sequence:
                    inputs_val = validation_dict['fc'].to(device)
                    # labels_val = validation_dict['true'][:, 0:100].long().to(device)
                    # trg_seq_val = validation_dict['true'][:, 0:100].long()

                    labels_val = validation_dict['true'][:, 10:100].long().to(device)
                    trg_seq_val = validation_dict['true'][:, 0:90].long()

                    # labels_val = validation_dict['true'][:, 0:100].long().to(device)
                    # trg_seq_val = validation_dict['pred'][:, 0:100].long()

                    # labels_val = validation_dict['true'][:, 10:100].long().to(device)
                    # trg_seq_val = validation_dict['pred'][:, 0:90].long()

                    trg_seq_val = random_replace(trg_seq_val, 0).long().to(device)
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
                data_sequence.mode = 'train'

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
                cm = cm / np.sum(cm, axis=1)[:, None]
                anot = np.sum(np.diag(cm)[1:6]) / 5 * 100
                vis.heatmap(X=cm, win='heatmap', opts=dict(title='confusion matrix',
                                                           rownames=['t0', 't1', 't2', 't3', 't4', 't5', 't_not'],
                                                           columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p_not']))
                vis.line(X=x, Y=np.column_stack((y_train_acc, y_batch_acc, anot)),
                         win='accuracy' + KF_Folders[fold_num], update='append',
                         opts=dict(title='accuracy' + KF_Folders[fold_num], showlegend=True,
                                   legend=['train', 'valid', 'valid_no_T']))
                vis.line(X=x, Y=np.column_stack((y_train_loss, y_batch_loss)),
                         win='loss' + KF_Folders[fold_num], update='append',
                         opts=dict(title='loss' + KF_Folders[fold_num], showlegend=True, legend=['train', 'valid']))

                # Save point
                if anot > best_accuracy:
                    best_accuracy = anot
                    if model_type == 'LSTM':
                        torch.save(model.state_dict(),
                                   './params/cross_validation/' + folders[fold_num] + '/params_LSTM_save_point.pkl')
                    else:
                        torch.save(model.state_dict(),
                                   './params/cross_validation/' + folders[fold_num] + '/params_trans_save_point.pkl')
                        # torch.save(model.state_dict(),
                        #            './params/cross_validation/' + folders[fold_num] + '/params_trans_90_noised.pkl')
                    print('the current best accuracy without transition phase is: %.3f %%' % anot)
                    txt = 'the current best accuracy is: %.3f%% at epoch %d, with' % (best_accuracy, epoch)
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


# fold_num = 6
model_type = 'LSTM'
# model_type = 'transformer'
for fold_num in range(0, 7):
    folders = ['div1', 'div2', 'div3', 'div4', 'div5', 'div6', 'div7']
    KF_Folders = ['folder1', 'folder2', 'folder3', 'folder4', 'folder5', 'folder6', 'folder7']
    main()
send_notice('training_finished', model_type)


