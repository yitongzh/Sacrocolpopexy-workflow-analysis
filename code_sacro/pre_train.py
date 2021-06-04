#! /usr/bin/env python
from model import Conv3_pre
from utils.UCF101 import UCF101
import numpy as np

import pickle
import time
import psutil
import os
import scipy.io as scio

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim


def dataloder():
    list_file = open('./data/UCF101-16-112-112/train_set1.pickle', 'rb')
    train_sets = pickle.load(list_file)
    list_file.close()

    list_file = open('./data/UCF101-16-112-112/train_set2.pickle', 'rb')
    train_sets.extend(pickle.load(list_file))
    list_file.close()

    list_file = open('./data/UCF101-16-112-112/train_labels1.pickle', 'rb')
    train_labels = pickle.load(list_file)
    list_file.close()

    list_file = open('./data/UCF101-16-112-112/train_labels2.pickle', 'rb')
    train_labels.extend(pickle.load(list_file))
    list_file.close()

    # info = psutil.virtual_memory()
    # print('memory used:', psutil.Process(os.getpid()).memory_info().rss)
    # print('total memory:', info.total)
    # print('precent used:', info.percent)
    return train_sets, train_labels


def reset_train_sets(train_sets, train_labels):
    myUCF101 = UCF101()
    batch_num = myUCF101.set_mode('train')
    print('reconstructing the train set...')
    for batch_index in range(batch_num):
        train_x, train_y = myUCF101[batch_index]
        train_sets[batch_index] = train_x
        train_labels[batch_index] = train_y

    return train_sets, train_labels


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Conv3_pre().to(device)

    print('loading data')
    start = time.time()
    train_sets, train_labels = dataloder()
    batch_num = len(train_sets)
    elapsed = (time.time() - start)
    print("Data loded, time used:", elapsed)

    # Initializing necessary components
    loss_func = nn.NLLLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    accuracy_stas = []
    loss_stas = []
    counter = 0
    train_idx = np.arange(batch_num)

    # myUCF101 = UCF101()
    # className = myUCF101.get_className()
    # batch_num = myUCF101.set_mode('train')

    print('Start training')
    start = time.time()
    for epoch in range(28):
        print('epoch:', epoch + 1)
        running_loss = 0.0
        running_accuracy = 0.0
        np.random.shuffle(train_idx)

        # reconstruct the train set for every 7 epochs
        if epoch % 7 == 0 and epoch != 0:
            train_sets, train_labels = reset_train_sets(train_sets, train_labels)

        for batch_idx in range(batch_num):
            inputs = train_sets[train_idx[batch_idx]].to(device)
            labels = train_labels[train_idx[batch_idx]].long().to(device)

            # inputs, labels = myUCF101[batch_idx]
            # inputs = inputs.to(device)
            # labels = labels.long().to(device)

            optimizer.zero_grad()
            output = model.forward_cov(inputs)
            loss = loss_func(output, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 10)
            optimizer.step()

            # Calculate the loss with new parameters
            running_loss += loss.item()
            current_loss = running_loss / (batch_idx + 1)

            # Calculate the accuracy
            output = model.forward_cov(inputs)
            _, predicted_labels = torch.max(output.data, 1)
            correct_pred = (predicted_labels == labels).sum().item()
            total_pred = predicted_labels.size(0)

            batch_accuracy = correct_pred / total_pred
            running_accuracy += batch_accuracy
            current_accuracy = running_accuracy / (batch_idx + 1) * 100

            counter += 1
            if counter % 10 == 0:
                # save the loss and accuracy
                accuracy_stas.append(current_accuracy)
                loss_stas.append(current_loss)

            if batch_idx % 300 == 299:
                print('[Epoch: %d Batch: %5d] loss: %.3f accuracy: %.3f'
                      % (epoch + 1, batch_idx + 1, current_loss, current_accuracy))

        print('[Final results of epoch: %d] loss: %.3f accuracy: %.3f'
              % (epoch + 1, current_loss, current_accuracy))

    elapsed = (time.time() - start)
    print("Training finished, time used:", elapsed / 60, 'min')

    torch.save(model.state_dict(), 'params_conv3.pkl')

    data_path = 'results_output.mat'
    scio.savemat(data_path, {'accuracy': np.asarray(accuracy_stas), 'loss': np.asarray(loss_stas)})


if __name__ == '__main__':
    main()
