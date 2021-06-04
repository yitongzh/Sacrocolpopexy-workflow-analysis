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


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Conv3_pre().to(device)
    model.load_state_dict(torch.load('params_conv3.pkl'))

    loss_func = nn.NLLLoss().to(device)

    myUCF101 = UCF101()
    className = myUCF101.get_className()
    batch_num = myUCF101.set_mode('test')

    running_loss = 0.0
    running_accuracy = 0.0

    example_inputs = []
    example_outputs = []

    for batch_idx in range(10):
        inputs, labels = myUCF101[batch_idx]
        ls1 = [className.index(label) for label in labels]

        inputs = inputs.to(device)
        labels = torch.LongTensor(ls1).to(device)

        output = model.forward_cov(inputs)
        loss = loss_func(output, labels)

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

        example_inputs.append(inputs.cpu().data.numpy())
        example_outputs.append(predicted_labels.cpu().numpy())

        print('[Epoch: %d Batch: %5d] loss: %.3f accuracy: %.3f'
              % (1, batch_idx + 1, current_loss, current_accuracy))

    # data_path = 'example_output.mat'
    # scio.savemat(data_path, {'Inputs': example_inputs, 'Outputs': example_outputs})


if __name__ == '__main__':
    main()
