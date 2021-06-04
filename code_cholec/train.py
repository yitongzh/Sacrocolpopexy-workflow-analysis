#! /usr/bin/env python
from model import Endo3D
from utils.cholec80_loder import cholec80_loder
import numpy as np

import pickle
import time
import psutil
import os
import scipy.io as scio
from tqdm import tqdm
import visdom

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.utils import data

from sklearn.metrics import confusion_matrix


def main():
    vis = visdom.Visdom(env='Endo3D')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Endo3D().to(device)

    # load pre-trained parameters
    Endo3D_state_dict = model.state_dict()
    # pre_state_dict = torch.load('./params/params_conv3.pkl')
    pre_state_dict = torch.load('./params/c3d.pickle')
    new_state_dict = {k: v for k, v in pre_state_dict.items() if k in Endo3D_state_dict}
    del new_state_dict['fc8.weight']
    del new_state_dict['fc8.bias']
    Endo3D_state_dict.update(new_state_dict)
    model.load_state_dict(Endo3D_state_dict)
    # model.load_state_dict(torch.load('./params/params_endo3d.pkl'))
    # model.load_state_dict(torch.load('./params/params_endo3d_save_point.pkl'))

    # Initializing necessary components
    w = torch.tensor([1.0, 1, 1, 1, 1, 1, 1])
    loss_func = nn.NLLLoss(weight=w).to(device)

    fc7_params = list(map(id, model.fc7.parameters()))
    fc8_params = list(map(id, model.fc8.parameters()))
    fcphase_params = list(map(id, model.fc_phase.parameters()))
    base_params = filter(lambda p: id(p) not in fc7_params + fc8_params + fcphase_params,
                         model.parameters())
    # optimizer = optim.Adam([{'params': base_params, 'lr': 1e-4},
    #                         {'params': model.fc7.parameters(), 'lr': 1e-3},
    #                         {'params': model.fc8.parameters(), 'lr': 1e-3},
    #                         {'params': model.fc_phase.parameters(), 'lr': 1e-3}])
    optimizer = optim.Adam([{'params': base_params, 'lr': 1e-5},
                            {'params': model.fc7.parameters(), 'lr': 1e-4},
                            {'params': model.fc8.parameters(), 'lr': 1e-4},
                            {'params': model.fc_phase.parameters(), 'lr': 1e-4}])

    # optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=3e-5)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6, weight_decay=3e-5)
    # optimizer = optim.SGD(model.parameters(), lr=1e-5, weight_decay=3e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.93)

    # loading the training and validation set
    print('loading data')
    start = time.time()
    evaluation_slot = 25
    rebuild_slot = 7
    cholec80 = cholec80_loder(train_epoch_size=5000, validation_epoch_size=600)
    # cholec80 = cholec80_loder(train_epoch_size=50, validation_epoch_size=60)
    train_loader = data.DataLoader(cholec80, 50, shuffle=True)
    valid_loader = data.DataLoader(cholec80, 60, shuffle=True)
    elapsed = (time.time() - start)
    print("Data loded, time used:", elapsed)

    accuracy_stas = []
    loss_stas = []
    counter = 0
    best_accuracy = 0

    print('Start training')
    start = time.time()
    for epoch in range(100):
        # reconstruct the train set for every 7 epochs
        if epoch % rebuild_slot == 0 and epoch != 0:
            cholec80.build_epoch()
            # sacro.build_validation()

        print('epoch:', epoch + 1)
        cholec80.mode = 'train'
        for labels, inputs in tqdm(train_loader, ncols=80):
            counter += 1
            model.train()
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            output, _ = model.forward_cov(inputs)
            train_loss = loss_func(output, labels)
            train_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            if counter % evaluation_slot == 0:
                # Evaluation on training set
                _, predicted_labels = torch.max(output.cpu().data, 1)
                correct_pred = (predicted_labels == labels.cpu()).sum().item()
                total_pred = predicted_labels.size(0)
                train_accuracy = correct_pred / total_pred * 100

                # visualization in visdom
                y_train_acc = torch.Tensor([train_accuracy])
                y_train_loss = torch.Tensor([train_loss.item()])


                # Evaluation on validation set
                model.eval()
                cholec80.mode = 'validation'
                running_loss = 0.0
                running_accuracy = 0.0
                valid_num = 0
                cm = np.zeros((7, 7))
                for labels_val, inputs_val in valid_loader:
                    inputs_val = inputs_val.float().to(device)
                    labels_val = labels_val.long().to(device)
                    output, _ = model.forward_cov(inputs_val)
                    valid_loss = loss_func(output, labels_val)

                    # Calculate the loss with new parameters
                    running_loss += valid_loss.item()
                    # current_loss = running_loss / (batch_counter + 1)

                    _, predicted_labels = torch.max(output.cpu().data, 1)
                    cm += confusion_matrix(labels_val.cpu().numpy(), predicted_labels, labels=[0, 1, 2, 3, 4, 5, 6])
                    correct_pred = (predicted_labels == labels_val.cpu()).sum().item()
                    total_pred = predicted_labels.size(0)
                    accuracy = correct_pred / total_pred
                    running_accuracy += accuracy
                    valid_num += 1
                    # current_accuracy = running_accuracy / (batch_counter + 1) * 100
                batch_loss = running_loss / valid_num
                batch_accuracy = running_accuracy / valid_num * 100
                cholec80.mode = 'train'

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
                vis.heatmap(X=cm, win='heatmap', opts=dict(title='confusion matrix',
                                                           rownames=['t1', 't2', 't3', 't4', 't5', 't6', 't7'],
                                                           columnnames=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']))
                vis.line(X=x, Y=np.column_stack((y_train_acc, y_batch_acc)), win='accuracy', update='append',
                         opts=dict(title='accuracy', showlegend=True, legend=['train', 'valid']))
                vis.line(X=x, Y=np.column_stack((y_train_loss, y_batch_loss)), win='loss', update='append',
                         opts=dict(title='loss', showlegend=True, legend=['train', 'valid']))

                # Save point
                if batch_accuracy > best_accuracy:
                    best_accuracy = batch_accuracy
                    torch.save(model.state_dict(), './params/params_endo3d_save_point.pkl')
                    print('the current best accuracy is: %.3f %%' % best_accuracy)
                    txt = 'the current best accuracy is: %.3f %%' % best_accuracy
                    vis.text(txt, win='cur_best', opts=dict(title='Current Best'))
                    vis.heatmap(X=cm, win='heatmap2', opts=dict(title='The current best confusion matrix',
                                                                rownames=['t1', 't2', 't3', 't4', 't5', 't6', 't7'],
                                                                columnnames=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']))
                # viz.image(img, opts=dict(title='Example input'))

        print('[Final results of epoch: %d] train loss: %.3f train accuracy: %.3f %%'
              ' validation loss: %.3f validation accuracy: %.3f %%'
              % (epoch + 1, train_loss, train_accuracy, batch_loss, batch_accuracy))
        # scheduler.step()
    elapsed = (time.time() - start)
    print("Training finished, time used:", elapsed / 60, 'min')

    torch.save(model.state_dict(), 'params_endo3d.pkl')

    data_path = 'results_output.mat'
    scio.savemat(data_path, {'accuracy': np.asarray(accuracy_stas), 'loss': np.asarray(loss_stas)})


if __name__ == '__main__':
    main()
