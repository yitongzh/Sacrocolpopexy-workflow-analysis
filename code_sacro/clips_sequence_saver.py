#! /usr/bin/env python
import numpy as np

import pickle
import sys, time
import os
import json
import codecs
import visdom

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim


from tqdm import tqdm
from model import Endo3D
from seq2seq_LSTM import seq2seq_LSTM
from transformer.transformer import Transformer
from utils.sequence_loder import sequence_loder
from utils.clip_sequence_loder import clip_sequence_loader


divs = ['div1', 'div2', 'div3', 'div4', 'div5', 'div6', 'div7']
counter = 0
save_path = '/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_sequence/whole/sequence_in_clips'
current_path = os.path.abspath(os.getcwd())
videos_path = os.path.join(current_path, 'data/sacro_jpg')
with open(os.path.join(videos_path, 'dataset_' + divs[counter] + '.json'), 'r') as json_data:
    temp = json.load(json_data)
video_list = temp['train'] + temp['validation'] + temp['test']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for video_name in video_list:
    print('The current video is:' + video_name)
    sacro = sequence_loder(divs[counter], device)
    sacro.whole_len_output(video_name)

    whole_labels = sacro.whole_labels
    whole_inputs = sacro.whole_inputs

    dirs = os.path.join(save_path, video_name)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    a = open(os.path.join(dirs, 'clip_seq.pickle'), 'wb')
    pickle.dump(whole_inputs, a)
    a.close()

    b = open(os.path.join(dirs, 'clip_seq_labels.pickle'), 'wb')
    pickle.dump(whole_labels, b)
    b.close()
