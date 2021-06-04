import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class seq2seq_LSTM(nn.Module):
    def __init__(self, output_dim=7, tar_seq_dim=7, embed_dim=1200, hidden_dim=1200, input_dim=1200, num_layers=1
                 , dropout=0.2, pad_idx=None):
        super(seq2seq_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tar_seq_dim = tar_seq_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        self.encoder = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.embedding = nn.Embedding(self.tar_seq_dim, self.embed_dim, padding_idx=pad_idx)
        self.decoder = nn.LSTM(self.embed_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)

        self.l1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, tar_seq):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        outputs, hidden = self.encoder(x)
        embedded = self.dropout(F.relu(self.embedding(tar_seq)))
        outputs, _ = self.decoder(embedded, hidden)
        prediction = self.softmax(self.l1(outputs))
        return prediction


class seq2seq_LSTM_merge(nn.Module):
    def __init__(self, output_dim=7, tar_seq_dim=7, clip_seq_dim=1000, embed_dim=600, hidden_dim=1200, input_dim=1200,
                 num_layers=1
                 , dropout=0.2):
        super(seq2seq_LSTM_merge, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tar_seq_dim = tar_seq_dim
        self.clip_seq_dim = clip_seq_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        self.encoder = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.tar_embedding = nn.Embedding(self.tar_seq_dim, self.embed_dim)
        self.clip_embedding = nn.Embedding(self.clip_seq_dim, self.embed_dim)
        self.decoder = nn.LSTM(self.embed_dim*2, self.hidden_dim, num_layers=self.num_layers, batch_first=True)


        self.l1 = nn.Linear(self.embed_dim * 2, self.output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, tar_seq, clip_num):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        outputs, hidden = self.encoder(x)
        embedded_tar = self.dropout(F.relu(self.tar_embedding(tar_seq)))
        embedded_clip = self.dropout(F.relu(self.clip_embedding(clip_num)))
        embedded = torch.cat((embedded_tar, embedded_clip), 2)
        outputs, _ = self.decoder(embedded, hidden)
        prediction = self.softmax(self.l1(outputs))
        return prediction
