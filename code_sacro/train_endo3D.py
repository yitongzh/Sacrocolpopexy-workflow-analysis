from many2many_LSTM import many2many_LSTM
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
import requests

from sklearn.metrics import confusion_matrix


class data_loder(data.Dataset):
    def __init__(self, video_name, max_len=2000):
        open_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_sequence/whole',
                                 KF_Folders[fold_num], video_name)

        file = open(os.path.join(open_path, 'seq_pred_c3d.pickle'), 'rb')
        self.seq_pre = pickle.load(file)
        file.close()

        file = open(os.path.join(open_path, 'seq_true.pickle'), 'rb')
        self.seq_true = pickle.load(file)
        file.close()

        file = open(os.path.join(open_path, 'fc_list.pickle'), 'rb')
        self.fc_list = pickle.load(file)
        file.close()

        self.seq_len = len(self.seq_true)
        current_path = os.path.abspath(os.getcwd())

        self.fcps = torch.cat((torch.stack(self.fc_list).cpu(), torch.zeros(max_len - self.seq_len, 1200)))
        self.fcs = torch.stack(self.fc_list).cpu()

    def __getitem__(self, idx):
        M = torch.ones(max_len, 1200)
        M[idx + 1:, :] = 0
        sequence_input_fc = self.fcps * M
        return {'fc': sequence_input_fc,
                'true': torch.tensor(self.seq_true[0:idx + 1]),
                'pred': torch.tensor(self.seq_pre[0: idx + 1])}

    def __len__(self):
        return self.seq_len


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


def evlauation(video_list, model, device):
    total_video_num = len(video_list)
    running_loss = 0
    running_accuracy = 0
    cm = np.zeros((7, 7))
    for video_name in tqdm(video_list, ncols=80):
        model.eval()
        video_loaded = data_loder(video_name)
        inputs = video_loaded.fcs.unsqueeze(0).to(device)
        labels = torch.tensor(video_loaded.seq_true).long().unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.forward(inputs)
        _, predicted_labels = torch.max(output.cpu().data, 2)
        # loss = sequence_loss(output[:, 0:len(video_loaded), :], labels, device)
        loss = sequence_loss(output, labels, device)
        predictions = predicted_labels[:, 0:len(video_loaded)]
        cm += confusion_matrix(labels.cpu().numpy().reshape(-1, ), predictions.view(-1, ), labels=[0, 1, 2, 3, 4, 5, 6])
        correct_pred = (predictions == labels.cpu()).sum().item()
        total_pred = predicted_labels.size(0) * predicted_labels.size(1)
        accuracy = correct_pred / total_pred * 100
        running_loss += loss.item()
        running_accuracy += accuracy
    running_loss = running_loss / total_video_num
    running_accuracy = running_accuracy / total_video_num
    cm = cm / np.sum(cm, axis=1)[:, None]
    return running_loss, running_accuracy, cm


def send_notice(event_name, text, key='d_kRy2XrftABFT4w0v7TeW'):
    report = {"value1": text}
    requests.post(f'https://maker.ifttt.com/trigger/{event_name}/with/key/{key}', data=report)


for fold_num in range(1, 2):
    print('fold_num:', fold_num)
    folders = ['div1', 'div2', 'div3', 'div4', 'div5', 'div6', 'div7']
    KF_Folders = ['folder1', 'folder2', 'folder3', 'folder4', 'folder5', 'folder6', 'folder7']

    current_path = os.path.abspath(os.getcwd())
    videos_path = os.path.join(current_path, 'data/sacro_jpg')
    with open(os.path.join(videos_path, 'dataset_' + folders[fold_num] + '.json'), 'r') as json_data:
        temp = json.load(json_data)
    train_video_list = temp['train']
    validation_video_list = temp['validation']

    vis = visdom.Visdom(env='Endo3D')
    print('Initializing the model')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = many2many_LSTM(hidden_dim=1200, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # , weight_decay=3e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

    counter = 0
    best_accuracy = 0
    folder_idx = 0
    rebuild_slot = 1
    evaluation_slot = 25
    batch_size = 10
    print('Start training')
    start = time.time()

    videos = [data_loder(video_name) for video_name in train_video_list]
    fcps = [video.fcps for video in videos]
    fcps = torch.stack(fcps)

    labels = [torch.tensor(video.seq_true).long().unsqueeze(0) for video in videos]

    for epoch in range(150):
        print('epoch:', epoch + 1)

        model.train()
        inputs = fcps.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        train_loss = [sequence_loss(output[i, 0:labels[i].size(1), :].unsqueeze(0),
                      labels[i].to(device), device) for i in range(len(train_video_list))]
        train_loss = sum(train_loss)/len(train_loss)
        train_loss.backward()
        optimizer.step()

        # random.shuffle(train_video_list)
        # for video_name in tqdm(train_video_list, ncols=80):
        #     video_loaded = data_loder(video_name)
        #     model.train()
        #     inputs = video_loaded.fcps.unsqueeze(0).to(device)
        #     labels = torch.tensor(video_loaded.seq_true).long().unsqueeze(0).to(device)
        #     optimizer.zero_grad()
        #     output = model.forward(inputs)
        #     # train_loss = sequence_loss(output, labels, device)
        #     train_loss = sequence_loss(output[:, 0:labels.size(1), :], labels, device)
        #     train_loss.backward()
        #     optimizer.step()

        # list_40 = list(range(40))
        # random.shuffle(list_40)
        # indices_list = [list_40[(i * batch_size):(i * batch_size + batch_size)] for i in range(int(40 / batch_size))]
        # for indices in tqdm(indices_list, ncols=80):
        #     video_batch = [videos[i] for i in indices]
        #     max_len = max([len(video) for video in video_batch])
        #     for t in range(max_len):
        #         model.train()
        #         inputs = torch.stack([video[t]['fc'] for video in video_batch]).to(device)
        #         labels = [video[t]['true'].unsqueeze(0).long().to(device) for video in video_batch]
        #         optimizer.zero_grad()
        #         output = model.forward(inputs)
        #         train_loss = [sequence_loss(output[i, 0:labels[i].size(1), :].unsqueeze(0), labels[i], device) for i in range(batch_size)]
        #         train_loss = sum(train_loss) / batch_size
        #         train_loss.backward()
        #         optimizer.step()

        # for video_name in tqdm(train_video_list, ncols=80):
        #     video_loaded = data_loder(video_name)
        #     for i in range(len(video_loaded)):
        #         model.train()
        #         inputs = video_loaded[i]['fc'].unsqueeze(0).to(device)
        #         labels = video_loaded[i]['true'].unsqueeze(0).long().to(device)
        #         optimizer.zero_grad()
        #         output = model.forward(inputs)
        #         train_loss = sequence_loss(output[:, 0:i+1, :], labels, device)
        #         train_loss.backward()
        #         optimizer.step()

        y_train_loss,  y_train_acc, _ = evlauation(train_video_list, model, device)
        y_batch_loss,  y_batch_acc, cm = evlauation(validation_video_list, model, device)


        # for video_name in tqdm(validation_video_list, ncols=80):
        #     video_loaded = data_loder(video_name)
        #     predictions = torch.zeros(1, len(video_loaded))
        #     for i in range(len(video_loaded)):
        #         model.eval()
        #         inputs = video_loaded[i]['fc'].to(device)
        #         with torch.no_grad():
        #             output = model.forward(inputs)
        #         _, predicted_labels = torch.max(output.cpu().data, 2)
        #         predictions[0, i] = predicted_labels[0, i]
        #     print(predictions)

        # # Evaluation on training set
        # _, predicted_labels = torch.max(output.cpu().data, 2)
        # correct_pred = (predicted_labels == labels.cpu()).sum().item()
        # total_pred = predicted_labels.size(0) * predicted_labels.size(1)
        # train_accuracy = correct_pred / total_pred * 100
        #
        # # visualization in visdom
        # y_train_acc = torch.Tensor([train_accuracy])
        # y_train_loss = torch.Tensor([train_loss.item()])
        #
        # # Evaluation on validation set
        # model.eval()
        # data_sequence.mode = 'validation'
        # running_loss = 0.0
        # running_accuracy = 0.0
        # valid_num = 0
        # cm = np.zeros((7, 7))
        # for validation_dict in data_sequence:
        #     inputs_val = validation_dict['fc'].to(device)
        #     labels_val = validation_dict['true'][:, 0:100].long().to(device)
        #
        #     with torch.no_grad():
        #         output = model.forward(inputs_val)
        #     valid_loss = sequence_loss(output, labels_val, device)
        #
        #     # Calculate the loss with new parameters
        #     running_loss += valid_loss.item()
        #     # current_loss = running_loss / (batch_counter + 1)
        #
        #     _, predicted_labels = torch.max(output.cpu().data, 2)
        #     cm += confusion_matrix(labels_val.cpu().numpy().reshape(-1, ), predicted_labels.view(-1, ),
        #                            labels=[0, 1, 2, 3, 4, 5, 6])
        #     correct_pred = (predicted_labels == labels_val.cpu()).sum().item()
        #     total_pred = predicted_labels.size(0) * predicted_labels.size(1)
        #     accuracy = correct_pred / total_pred
        #     running_accuracy += accuracy
        #     valid_num += 1
        #     # current_accuracy = running_accuracy / (batch_counter + 1) * 100
        # batch_loss = running_loss / valid_num
        # batch_accuracy = running_accuracy / valid_num * 100
        # data_sequence.mode = 'train'
        #
        # # save the loss and accuracy
        # accuracy_stas.append(batch_accuracy)
        # loss_stas.append(batch_loss)
        #
        # # visualization in visdom
        # x = torch.Tensor([counter])
        # y_batch_acc = torch.Tensor([batch_accuracy])
        # y_batch_loss = torch.Tensor([batch_loss])
        # txt1 = ''.join(['t%d:%d ' % (i, np.sum(cm, axis=1)[i]) for i in range(len(np.sum(cm, axis=1)))])
        # txt2 = ''.join(['p%d:%d ' % (i, np.sum(cm, axis=0)[i]) for i in range(len(np.sum(cm, axis=0)))])
        # vis.text((txt1 + '<br>' + txt2), win='summary', opts=dict(title='Summary'))
        anot = np.sum(np.diag(cm)[1:6]) / 5 * 100
        x = torch.Tensor([epoch])
        vis.heatmap(X=cm, win='heatmap', opts=dict(title='confusion matrix',
                                                   rownames=['t0', 't1', 't2', 't3', 't4', 't5', 't_not'],
                                                   columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p_not']))
        vis.line(X=x, Y=np.column_stack((y_train_acc, y_batch_acc)), win='accuracy', update='append',
                 opts=dict(title='accuracy', showlegend=True, legend=['train', 'valid', 'valid_no_T']))
        vis.line(X=x, Y=np.column_stack((y_train_loss, y_batch_loss)), win='loss', update='append',
                 opts=dict(title='loss', showlegend=True, legend=['train', 'valid']))

        # Save point
        if anot > best_accuracy:
            x = list(np.diag(cm) * 100)
            best_accuracy = anot
            # torch.save(model.state_dict(), './params/params_LSTM_save_point.pkl')
            torch.save(model.state_dict(),
                       './params/cross_validation/' + folders[fold_num] + '/params_LSTM_e3d.pkl')
            print('the current best accuracy without transition phase is: %.3f %%' % best_accuracy)
            txt = 'the current best accuracy is: %.3f%%, with' % best_accuracy
            vis.text(txt, win='cur_best', opts=dict(title='Current Best'))
            vis.heatmap(X=cm, win='heatmap1', opts=dict(title='best confusion matrix',
                                                       rownames=['t0', 't1', 't2', 't3', 't4', 't5', 't_not'],
                                                       columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p_not']))
            # txt3 = ''.join(['Phase %d accuracy:%d%%   ' % (i + 1, x[i]) for i in range(len(x))])
            # vis.text((txt + '<br>' + txt3), win='cur_best', opts=dict(title='Current Best'))
        # viz.image(img, opts=dict(title='Example input'))

        scheduler.step()
    elapsed = (time.time() - start)
    print("Training finished, time used:", elapsed / 60, 'min')
send_notice('training_finished', "e3d")




