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
from model import Endo3D, naive_end2end
from many2many_LSTM import many2many_LSTM
from transformer.transformer import Transformer
from utils.clip_sequence_loder import clip_sequence_loader
from sklearn.metrics import confusion_matrix


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
    return trg_seq.long()


def main():
    def sequence_loss(output, labels):
        # alpha = np.array([28.33, 11.23, 5.00, 2.09, 36.36, 11.01, 5.98]) / 100
        # loss_func = FocalLoss(7, alpha=alpha).to(device)
        # w = [0.1, 1.6, 0.5, 0.8, 2.0, 1, 1]
        # w = torch.tensor([0.1, 1.5, 0.3, 0.6, 2.0, 1, 1])
        output = output.cuda()
        labels = labels.cuda()
        w = torch.tensor([0.01, 1, 1, 1, 1, 1, 1])
        loss_func = nn.NLLLoss(weight=w).cuda()
        l = output.size()[1]
        pred_labels = [torch.max(output[:, i, :].data, 1)[1] for i in range(l - 1)]
        pred_labels.insert(0, torch.max(output[:, 0, :].data, 1)[1])
        loss_cont = sum([loss_func(output[:, i, :], pred_labels[i]) for i in range(l)]) / l
        loss = sum([loss_func(output[:, i, :], labels[:, i]) for i in range(l)]) / l
        return (loss + 0 * loss_cont).cuda()

    # class end2end(nn.Module):
    #     def __init__(self, model2, insight_length=100):
    #         super(end2end, self).__init__()
    #         self.insight_length = insight_length
    #         self.model1 = Endo3D()
    #         self.model2 = model2
    #
    #     def forward(self, x):
    #         # batch_size = tar_seq.size(0)
    #         fc8s = self.model1.forward_cov(x[0].cuda())[1].unsqueeze(1)
    #         for i in range(self.insight_length - 1):
    #             fc8s = torch.cat((fc8s, self.model1.forward_cov(x[i + 1].cuda())[1].unsqueeze(1)), 1)
    #         output_ = self.model2.forward(fc8s)
    #         return output_

    class end2end(nn.Module):
        def __init__(self, model2, insight_length=100):
            super(end2end, self).__init__()
            self.insight_length = insight_length
            self.model1 = Endo3D()
            self.model2 = model2

        def forward(self, x):
            batch_size = x[0].size(0)
            input_ = torch.zeros(batch_size, self.insight_length, 3, 16, 112, 112)
            for i in range(self.insight_length):
                input_[:, i, :, :, :, :] = x[i]
            input_ = input_.view(batch_size * self.insight_length, 3, 16, 112, 112)

            fc8s = self.model1.forward_cov(input_.cuda())[1]
            fc8s = fc8s.view(batch_size, self.insight_length, 1200)
            # for i in range(self.insight_length - 1):
            #     fc8s = torch.cat((fc8s, self.model1.forward_cov(x[i + 1].cuda())[1].unsqueeze(1)), 1)
            output_ = self.model2.forward(fc8s)
            return output_

    div = 'div2'
    print('The current folder is:', div)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # seq_model_type = 'transformer'
    seq_model_type = 'LSTM'
    number_of_validation_per_epoch = 2

    print("Initializing the model...")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if seq_model_type == 'LSTM':
        sequential_model = many2many_LSTM(hidden_dim=1200, num_layers=3)
        vis = visdom.Visdom(env='LSTM')
        model = end2end(sequential_model)
        model.model1.load_state_dict(torch.load('./params/cross_validation/' + div + '/params_endo3d.pkl',
                                                map_location='cuda:0'))
        model.model2.load_state_dict(torch.load('./params/cross_validation/' + div + '/params_LSTM_m2m.pkl',
                                                map_location='cuda:0'))
        # model = naive_end2end()

    print('The sequential model is :', seq_model_type)
    lr = 1e-8
    # optimizer = optim.Adam([{'params': model.model1.parameters(), 'lr': lr * 0.1},
    #                         {'params': model.model2.parameters(), 'lr': lr}])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=3e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
    model = nn.DataParallel(model)
    model = model.cuda()
    print("Model initialized.")

    current_path = os.path.abspath(os.getcwd())
    videos_path = os.path.join(current_path, 'data/sacro_jpg')
    with open(os.path.join(videos_path, 'dataset_' + div + '.json'), 'r') as json_data:
        temp = json.load(json_data)
    train_list = temp['train']
    validation_list = temp['validation']
    # sacro_validation_list = [clip_sequence_loader([validation_list[0]], is_augmentation=False),
    #                          clip_sequence_loader([validation_list[1]], is_augmentation=False)]
    sacro_validation = clip_sequence_loader(validation_list[0:2], is_augmentation=False)
    best_accuracy = 0

    if True:
        # Evaluate the model on the validation set
        cm = np.zeros((7, 7))
        running_loss = 0.0
        running_accuracy = 0.0
        for i in tqdm(range(len(sacro_validation)), ncols=80):
            x_vali = sacro_validation[i]
            inputs_vali = x_vali['inputs']
            tar_seq_vali = x_vali['labels'][:, 0:90]
            # tar_seq_vali = torch.zeros(inputs_vali[0].size(0), 100).long()
            labels_vali = x_vali['labels']
            # tar_seq_vali = random_replace(tar_seq_vali, 0.4)

            with torch.no_grad():
                output_vali = model(inputs_vali)
            _, predicted_labels = torch.max(output_vali.cpu().data, 2)
            cm += confusion_matrix(labels_vali.numpy().reshape(-1, ), predicted_labels.view(-1, ),
                                   labels=[0, 1, 2, 3, 4, 5, 6])
            correct_pred = (predicted_labels == labels_vali).sum().item()
            total_pred = predicted_labels.size(0) * predicted_labels.size(1)
            running_accuracy += correct_pred / total_pred
            running_loss += sequence_loss(output_vali, labels_vali).item()
            torch.cuda.empty_cache()
        average_accuracy = running_accuracy / len(sacro_validation) * 100
        average_loss = running_loss / len(sacro_validation)

        # visualization in visdom
        x = torch.Tensor([0])
        y_batch_acc = torch.Tensor([average_accuracy])
        y_batch_loss = torch.Tensor([average_loss])
        y_train_acc = torch.Tensor([0])
        y_train_loss = torch.Tensor([0])
        txt1 = ''.join(['t%d:%d ' % (i, np.sum(cm, axis=1)[i]) for i in range(len(np.sum(cm, axis=1)))])
        txt2 = ''.join(['p%d:%d ' % (i, np.sum(cm, axis=0)[i]) for i in range(len(np.sum(cm, axis=0)))])
        vis.text((txt1 + '<br>' + txt2), win='summary', opts=dict(title='Summary'))
        cm = cm / np.sum(cm, axis=1)[:, None] * 100
        anot = np.sum(np.diag(cm)[1:6]) / 5
        vis.heatmap(X=cm, win='heatmap', opts=dict(title='confusion matrix',
                                                   rownames=['t0', 't1', 't2', 't3', 't4', 't5', 't_not'],
                                                   columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p_not']))
        vis.line(X=x, Y=np.column_stack((y_train_acc, y_batch_acc, anot)), win='accuracy', update='append',
                 opts=dict(title='accuracy', showlegend=True, legend=['train', 'valid', 'valid_no_T']))
        vis.line(X=x, Y=np.column_stack((y_train_loss, y_batch_loss)), win='loss', update='append',
                 opts=dict(title='loss', showlegend=True, legend=['train', 'valid']))

    # sacro_ls = [clip_sequence_loader([train_list[i]], is_augmentation=False) for i in range(len(train_list))]

    running_counter = 0
    for epoch in range(100):
        # Train the model
        print("Epoch:", epoch, "Loading the training data...")

        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        print('Current learning rates are:', learning_rate)
        print('Current learning rates are:', learning_rate)
        random.shuffle(train_list)
        sacro1 = clip_sequence_loader(train_list[0:4], is_augmentation=True)
        sacro2 = clip_sequence_loader(train_list[4:8], is_augmentation=True)
        sacro3 = clip_sequence_loader(train_list[8:], is_augmentation=True)
        sacro_ls = [sacro1, sacro2, sacro3]
        print("Training data loaded")

        seq_counter = 0
        for i in tqdm(range(len(sacro_ls[0])), ncols=80):
            random.shuffle(sacro_ls)
            for sacro in sacro_ls:
                x = sacro[i]
                inputs = x['inputs']
                # tar_seq = x['labels'][:, 0:90]
                # tar_seq = torch.zeros(inputs[0].size(0), 100).long()
                labels = x['labels']
                optimizer.zero_grad()
                # tar_seq = random_replace(tar_seq, 0.4)
                output = model(inputs)
                # _, predicted_labels = torch.max(output.cpu().data, 2)

                train_loss = sequence_loss(output, labels)
                train_loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            seq_counter += 1
            if seq_counter == (200 / number_of_validation_per_epoch):
                seq_counter = 0
                # Evaluation on training set
                _, predicted_labels = torch.max(output.cpu().data, 2)
                correct_pred = (predicted_labels == labels).sum().item()
                total_pred = predicted_labels.size(0) * predicted_labels.size(1)
                train_accuracy = correct_pred / total_pred * 100
                y_train_acc = torch.Tensor([train_accuracy])
                y_train_loss = torch.Tensor([train_loss.item()])

                # Evaluate the model on the validation set
                cm = np.zeros((7, 7))
                running_loss = 0.0
                running_accuracy = 0.0
                for i in tqdm(range(len(sacro_validation)), ncols=80):
                    x_vali = sacro_validation[i]
                    inputs_vali = x_vali['inputs']

                    labels_vali = x_vali['labels']
                    with torch.no_grad():
                        output_vali = model(inputs_vali)
                    _, predicted_labels = torch.max(output_vali.cpu().data, 2)
                    cm += confusion_matrix(labels_vali.numpy().reshape(-1, ), predicted_labels.view(-1, ),
                                           labels=[0, 1, 2, 3, 4, 5, 6])
                    correct_pred = (predicted_labels == labels_vali).sum().item()
                    total_pred = predicted_labels.size(0) * predicted_labels.size(1)
                    running_accuracy += correct_pred / total_pred
                    running_loss += sequence_loss(output_vali, labels_vali).item()
                    torch.cuda.empty_cache()

                average_accuracy = running_accuracy / (len(sacro_validation)) * 100
                average_loss = running_loss / (len(sacro_validation))

                # visualization in visdom
                x = torch.Tensor([running_counter / number_of_validation_per_epoch + 0.5])
                y_batch_acc = torch.Tensor([average_accuracy])
                y_batch_loss = torch.Tensor([average_loss])

                txt1 = ''.join(['t%d:%d ' % (i, np.sum(cm, axis=1)[i]) for i in range(len(np.sum(cm, axis=1)))])
                txt2 = ''.join(['p%d:%d ' % (i, np.sum(cm, axis=0)[i]) for i in range(len(np.sum(cm, axis=0)))])
                vis.text((txt1 + '<br>' + txt2), win='summary', opts=dict(title='Summary'))
                cm = cm / np.sum(cm, axis=1)[:, None] * 100
                anot = np.sum(np.diag(cm)[1:6]) / 5
                vis.heatmap(X=cm, win='heatmap', opts=dict(title='confusion matrix',
                                                           rownames=['t0', 't1', 't2', 't3', 't4', 't5', 't_not'],
                                                           columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p_not']))
                vis.line(X=x, Y=np.column_stack((y_train_acc, y_batch_acc, anot)), win='accuracy', update='append',
                         opts=dict(title='accuracy', showlegend=True, legend=['train', 'valid', 'valid_no_T']))
                vis.line(X=x, Y=np.column_stack((y_train_loss, y_batch_loss)), win='loss', update='append',
                         opts=dict(title='loss', showlegend=True, legend=['train', 'valid']))
                running_counter += 1

                # Save point
                if anot > best_accuracy:
                    best_accuracy = anot
                    if seq_model_type == 'LSTM':
                        torch.save(model.module.state_dict(), './params/params_LSTM_save_point.pkl')
                    else:
                        torch.save(model.module.state_dict(), './params/params_trans_save_point.pkl')
                    # print('the current best accuracy without transition phase is: %.3f %%' % anot)
                    txt = 'the current best accuracy is: %.3f%%, with' % best_accuracy
                    x = list(np.diag(cm)[1:6])
                    txt3 = ''.join(['Phase %d accuracy:%d%%   ' % (i + 1, x[i]) for i in range(len(x))])
                    vis.text((txt + '<br>' + txt3), win='cur_best', opts=dict(title='Current Best'))
    scheduler.step()


if __name__ == '__main__':
    main()
