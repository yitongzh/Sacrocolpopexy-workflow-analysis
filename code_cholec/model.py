#! /usr/bin/env python
import torch
import torch.nn as nn
import torch.autograd as autograd
import math
from functools import partial
from torch.autograd import Variable
import numpy as np
import torch.optim as optim


class Conv3_pre(nn.Module):
    def __init__(self):
        super(Endo3D, self).__init__()

        self.losses = []
        self.accuracies = []

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc_class = nn.Linear(4096, 101)

        # self.fc_phase = nn.Linear(4096 + 7, 5)
        #
        # self.lstm1 = nn.LSTM(4096 + 7, 4096)
        # self.lstm2 = nn.LSTM(4096, 4096)
        # self.lstm3 = nn.LSTM(4096, 128)
        # self.lstm_phase = nn.Linear(128, 5)

        self.relu = nn.ReLU()
        self.drop_layer = nn.Dropout(p=0.5)
        self.pool_122 = nn.MaxPool3d((1, 2, 2))
        self.pool_222 = nn.MaxPool3d((2, 2, 2))
        self.pool_222_ceil = nn.MaxPool3d((2, 2, 2), ceil_mode=True)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.hidden1 = self.init_hidden(4096)
        # self.hidden2 = self.init_hidden(4096)
        # self.hidden3 = self.init_hidden(128)

    #     def _make_conv_layer(self, in_c, out_c):
    #         conv_layer = nn.Sequential(
    #             nn.Conv3d(in_c, in_c * 2, kernel_size=(3, 3, 3), padding=1),
    #             nn.ReLU(),
    #             nn.Conv3d(in_c * 2, out_c, kernel_size=(3, 3, 3), padding=1),
    #             nn.MaxPool3d((2, 2, 2)),
    #         )
    #         return conv_layer

    def forward_cov(self, x):
        x = self.pool_122(self.conv1(x))
        x = self.relu(x)
        # print(x.size())
        x = self.pool_222(self.conv2(x))
        x = self.relu(x)
        # print(x.size())
        x = self.conv3a(x)
        # x = self.drop_layer(x)
        x = self.relu(x)
        # print(x.size())
        x = self.pool_222(self.conv3b(x))
        x = self.relu(x)
        # print(x.size())
        x = self.conv4a(x)
        # x = self.drop_layer(x)
        x = self.relu(x)
        # print(x.size())
        x = self.pool_222(self.conv4b(x))
        x = self.relu(x)
        # print(x.size())
        x = self.conv5a(x)
        # x = self.drop_layer(x)
        x = self.relu(x)
        # print(x.size())
        x = self.pool_222_ceil(self.conv5b(x))
        x = self.relu(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc6(x)
        x = self.drop_layer(x)
        x = self.relu(x)
        # print(x.size())
        x = self.fc7(x)
        x = self.drop_layer(x)
        x = self.relu(x)
        # print(x.size())
        output = self.softmax(self.fc_class(x))

        return output


class Endo3D(nn.Module):
    def __init__(self):
        super(Endo3D, self).__init__()

        self.losses = []
        self.accuracies = []

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)

        # for p in self.parameters():
        #     p.requires_grad = False

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1200)
        # self.fc_class = nn.Linear(4096, 101)
        self.fc_phase = nn.Linear(1200, 7)

        self.relu = nn.ReLU()

        self.pool_122 = nn.MaxPool3d((1, 2, 2))
        self.pool_222 = nn.MaxPool3d((2, 2, 2))
        self.pool_222_ceil = nn.MaxPool3d((2, 2, 2), ceil_mode=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.drop_layer = nn.Dropout(p=0.5)
        # self.hidden1 = self.init_hidden(4096)
        # self.hidden2 = self.init_hidden(4096)
        # self.hidden3 = self.init_hidden(128)

        # for p in self.parameters():
        #     p.requires_grad = False

    def forward_cov(self, x):
        x = self.pool_122(self.conv1(x))
        x = self.relu(x)
        # print(x.size())
        x = self.pool_222(self.conv2(x))
        x = self.relu(x)
        # print(x.size())
        x = self.conv3a(x)
        x = self.relu(x)
        # print(x.size())
        x = self.pool_222(self.conv3b(x))
        x = self.relu(x)
        # x = self.drop_layer(x)
        # print(x.size())
        x = self.conv4a(x)
        x = self.relu(x)
        # print(x.size())
        x = self.pool_222(self.conv4b(x))
        x = self.relu(x)
        # x = self.drop_layer(x)
        # print(x.size())
        x = self.conv5a(x)
        x = self.relu(x)
        # print(x.size())
        x = self.pool_222_ceil(self.conv5b(x))
        x = self.relu(x)
        # x = self.drop_layer(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.relu(self.fc6(x))
        x = self.drop_layer(x)
        # print(x.size())
        x = self.relu(self.fc7(x))
        x = self.drop_layer(x)
        x = self.relu(self.fc8(x))
        x = self.drop_layer(x)
        x_fc = x
        x = self.fc_phase(x)
        output = self.softmax(x)

        return output, x_fc


class Endo3D_1vo(nn.Module):
    def __init__(self):
        super(Endo3D_1vo, self).__init__()

        self.losses = []
        self.accuracies = []

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)

        for p in self.parameters():
            p.requires_grad = False

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.cl0 = self._make_layer(4096, 1)
        self.cl1 = self._make_layer(4096, 1)
        self.cl2 = self._make_layer(4096, 1)
        self.cl3 = self._make_layer(4096, 1)
        self.cl4 = self._make_layer(4096, 1)
        self.cl5 = self._make_layer(4096, 1)

        self.relu = nn.ReLU()
        self.pool_122 = nn.MaxPool3d((1, 2, 2))
        self.pool_222 = nn.MaxPool3d((2, 2, 2))
        self.pool_222_ceil = nn.MaxPool3d((2, 2, 2), ceil_mode=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.drop_layer = nn.Dropout(p=0.5)

    def _make_layer(self, cin, cout):
        fc_layers = nn.Sequential(
            nn.Linear(cin, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(5, cout))
        return fc_layers

    def forward_cov(self, x):
        x = self.pool_122(self.conv1(x))
        x = self.relu(x)
        x = self.pool_222(self.conv2(x))
        x = self.relu(x)
        x = self.conv3a(x)
        x = self.relu(x)
        x = self.pool_222(self.conv3b(x))
        x = self.relu(x)
        x = self.conv4a(x)
        x = self.relu(x)
        x = self.pool_222(self.conv4b(x))
        x = self.relu(x)
        x = self.conv5a(x)
        x = self.relu(x)
        x = self.pool_222_ceil(self.conv5b(x))
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.drop_layer(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.drop_layer(x)

        x0 = self.cl0(x)
        x1 = self.cl1(x)
        x2 = self.cl2(x)
        x3 = self.cl3(x)
        x4 = self.cl4(x)
        x5 = self.cl5(x)
        x = torch.cat((x0, x1, x2, x3, x4, x5), 1)
        output = self.softmax(x)
        return output


class Endo3D_1vo1(nn.Module):
    def __init__(self):
        super(Endo3D_1vo1, self).__init__()

        self.losses = []
        self.accuracies = []

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)

        for p in self.parameters():
            p.requires_grad = False

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.cl0 = self._make_layer(4096, 400)
        self.cl1 = self._make_layer(4096, 400)
        self.cl2 = self._make_layer(4096, 400)
        self.cl3 = self._make_layer(4096, 400)
        self.cl4 = self._make_layer(4096, 400)
        self.cl5 = self._make_layer(4096, 400)

        self.fc8 = nn.Linear(2400+4096, 1200)
        self.fc_ph = nn.Linear(1200, 6)

        self.relu = nn.ReLU()
        self.pool_122 = nn.MaxPool3d((1, 2, 2))
        self.pool_222 = nn.MaxPool3d((2, 2, 2))
        self.pool_222_ceil = nn.MaxPool3d((2, 2, 2), ceil_mode=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.drop_layer = nn.Dropout(p=0.5)

    def _make_layer(self, cin, cout):
        fc_layers = nn.Sequential(
            nn.Linear(cin, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, cout),
            nn.ReLU())
        return fc_layers

    def forward_cov(self, x):
        x = self.pool_122(self.conv1(x))
        x = self.relu(x)
        x = self.pool_222(self.conv2(x))
        x = self.relu(x)
        x = self.conv3a(x)
        x = self.relu(x)
        x = self.pool_222(self.conv3b(x))
        x = self.relu(x)
        x = self.drop_layer(x)
        x = self.conv4a(x)
        x = self.relu(x)
        x = self.pool_222(self.conv4b(x))
        x = self.relu(x)
        x = self.drop_layer(x)
        x = self.conv5a(x)
        x = self.relu(x)
        x = self.pool_222_ceil(self.conv5b(x))
        x = self.relu(x)
        x = self.drop_layer(x)

        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.drop_layer(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.drop_layer(x)

        x0 = self.cl0(x)
        x1 = self.cl1(x)
        x2 = self.cl2(x)
        x3 = self.cl3(x)
        x4 = self.cl4(x)
        x5 = self.cl5(x)
        x = torch.cat((x, x0, x1, x2, x3, x4, x5), 1)

        x = self.fc8(x)
        # x = self.relu(x)
        x = self.drop_layer(x)
        x = self.fc_ph(x)
        output = self.softmax(x)
        return output


class Endo3D_1vo2(nn.Module):
    def __init__(self):
        super(Endo3D_1vo2, self).__init__()

        self.losses = []
        self.accuracies = []

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.cl0 = self._make_layer(4096, 1)
        self.cl1 = self._make_layer(4096, 1)
        self.cl2 = self._make_layer(4096, 1)
        self.cl3 = self._make_layer(4096, 1)
        self.cl4 = self._make_layer(4096, 1)
        self.cl5 = self._make_layer(4096, 1)

        for p in self.parameters():
            p.requires_grad = False

        self.fc8 = nn.Linear(6, 50)
        self.fc9 = nn.Linear(50, 100)
        self.fc10 = nn.Linear(100, 1000)
        self.fc11 = nn.Linear(1000, 4096)
        self.fc12 = nn.Linear(4096, 6)

        self.relu = nn.ReLU()
        self.pool_122 = nn.MaxPool3d((1, 2, 2))
        self.pool_222 = nn.MaxPool3d((2, 2, 2))
        self.pool_222_ceil = nn.MaxPool3d((2, 2, 2), ceil_mode=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.drop_layer = nn.Dropout(p=0.5)

    def _make_layer(self, cin, cout):
        fc_layers = nn.Sequential(
            nn.Linear(cin, 2048),
            nn.ReLU(),
            nn.Dropout(p=0),
            nn.Linear(2048, 5),
            nn.ReLU(),
            nn.Dropout(p=0),
            nn.Linear(5, cout))
        return fc_layers

    def forward_cov(self, x):
        x = self.pool_122(self.conv1(x))
        x = self.relu(x)
        x = self.pool_222(self.conv2(x))
        x = self.relu(x)
        x = self.conv3a(x)
        x = self.relu(x)
        x = self.pool_222(self.conv3b(x))
        x = self.relu(x)
        x = self.conv4a(x)
        x = self.relu(x)
        x = self.pool_222(self.conv4b(x))
        x = self.relu(x)
        x = self.conv5a(x)
        x = self.relu(x)
        x = self.pool_222_ceil(self.conv5b(x))
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.relu(x)
        # x = self.drop_layer(x)
        x = self.fc7(x)
        x = self.relu(x)
        # x = self.drop_layer(x)

        x0 = self.cl0(x)
        x1 = self.cl1(x)
        x2 = self.cl2(x)
        x3 = self.cl3(x)
        x4 = self.cl4(x)
        x5 = self.cl5(x)
        x = torch.cat((x0, x1, x2, x3, x4, x5), 1)

        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.relu(self.fc10(x))
        x = self.relu(self.fc11(x))
        x = self.fc12(x)
        output = self.softmax(x)
        return output


class Endo3D_for_sequence(nn.Module): # based on 1vo1
    def __init__(self):
        super(Endo3D_for_sequence, self).__init__()

        self.losses = []
        self.accuracies = []

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1200)

        self.relu = nn.ReLU()

        self.pool_122 = nn.MaxPool3d((1, 2, 2))
        self.pool_222 = nn.MaxPool3d((2, 2, 2))
        self.pool_222_ceil = nn.MaxPool3d((2, 2, 2), ceil_mode=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.drop_layer = nn.Dropout(p=0.5)

        for p in self.parameters():
            p.requires_grad = False


    def forward_cov(self, x):
        x = self.pool_122(self.conv1(x))
        x = self.relu(x)
        # print(x.size())
        x = self.pool_222(self.conv2(x))
        x = self.relu(x)
        # print(x.size())
        x = self.conv3a(x)
        x = self.relu(x)
        # print(x.size())
        x = self.pool_222(self.conv3b(x))
        x = self.relu(x)
        # x = self.drop_layer(x)
        # print(x.size())
        x = self.conv4a(x)
        x = self.relu(x)
        # print(x.size())
        x = self.pool_222(self.conv4b(x))
        x = self.relu(x)
        # x = self.drop_layer(x)
        # print(x.size())
        x = self.conv5a(x)
        x = self.relu(x)
        # print(x.size())
        x = self.pool_222_ceil(self.conv5b(x))
        x = self.relu(x)
        # x = self.drop_layer(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.relu(self.fc6(x))
        x = self.drop_layer(x)
        # print(x.size())
        x = self.relu(self.fc7(x))
        x = self.drop_layer(x)
        output = self.relu(self.fc8(x))
        return output


class naive_end2end(nn.Module):
    def __init__(self, insight_length=100):
        super(naive_end2end, self).__init__()
        self.insight_length = insight_length

        self.losses = []
        self.accuracies = []

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 2048)
        self.fc8 = nn.Linear(2048, 512)

        self.relu = nn.ReLU()
        self.drop_layer = nn.Dropout(p=0.5)
        self.pool_122 = nn.MaxPool3d((1, 2, 2))
        self.pool_222 = nn.MaxPool3d((2, 2, 2))
        self.pool_222_ceil = nn.MaxPool3d((2, 2, 2), ceil_mode=True)
        self.softmax = nn.LogSoftmax(dim=2)

        self.encoder = nn.LSTM(512, 512, num_layers=3, batch_first=True)
        self.l1 = nn.Linear(512, 7)

    def forward(self, x):
        batch_size = x[0].size(0)
        input_ = torch.zeros(batch_size, self.insight_length, 3, 16, 112, 112)
        for i in range(self.insight_length):
            input_[:, i, :, :, :, :] = x[i]
        input_ = input_.view(batch_size * self.insight_length, 3, 16, 112, 112)

        x = self.pool_122(self.conv1(input_.cuda()))
        x = self.relu(x)
        x = self.pool_222(self.conv2(x))
        x = self.relu(x)
        x = self.conv3a(x)
        x = self.relu(x)
        x = self.pool_222(self.conv3b(x))
        x = self.relu(x)
        x = self.conv4a(x)
        x = self.relu(x)
        x = self.pool_222(self.conv4b(x))
        x = self.relu(x)
        x = self.conv5a(x)
        x = self.relu(x)
        x = self.pool_222_ceil(self.conv5b(x))
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc6(x))
        x = self.drop_layer(x)
        x = self.relu(self.fc7(x))
        x = self.drop_layer(x)
        fc8s = self.relu(self.fc8(x))
        fc8s = fc8s.view(batch_size, self.insight_length, 512)

        self.encoder.flatten_parameters()
        outputs, hidden = self.encoder(fc8s)
        output_ = self.softmax(self.l1(outputs))

        return output_


