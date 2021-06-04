import sys
from model import Endo3D
from seq2seq_LSTM import seq2seq_LSTM
from transformer.transformer import Transformer
from utils.sequence_loder import sequence_loder
import numpy as np
import visdom
from tqdm import tqdm
import cv2

import time
import os
import scipy.io as scio
from scipy import stats
import random

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
from torch.utils import data
import pickle
import json

from sklearn.metrics import confusion_matrix


def mode_average(prediction, slinwin_sz=17):
    results = []
    hf = int((slinwin_sz - 1) / 2)
    prediction_pad = np.hstack((np.full([hf, ], np.nan), prediction))
    prediction_pad = np.hstack((prediction_pad, np.full([hf, ], np.nan)))
    for i in range(prediction.shape[0]):
        st_idx = i
        ed_idx = i + slinwin_sz - 1
        results.append(stats.mode(prediction_pad[st_idx:ed_idx])[0][0])
    return np.array(results)


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


def augmentation(clip):
    clip_temp = clip.transpose((1, 2, 3, 0))
    method = 'non'
    for i in range(16):
        img = clip_temp[i, :, :, :].astype('uint8')
        if method != 'non':
            if method == 'random_flip':
                flip_num = random.choice([-1, 0, 1])
                img = cv2.flip(img, flip_num)
            elif method == 'random_rot':
                rows, cols, _ = np.shape(img)
                rot_angle = random.random() * 360
                M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), rot_angle, 1)
                img = cv2.warpAffine(img, M, (cols, rows))
            elif method == 'crop':
                rows, cols, _ = np.shape(img)
                result_size = random.randint(30, 56)
                result_row = random.randint(result_size, rows - result_size)
                result_col = random.randint(result_size, cols - result_size)
                img = img[result_row - result_size:result_row + result_size,
                      result_col - result_size:result_col + result_size, :]
                img = cv2.resize(img, (112, 112))
            elif method == 'Gauss_filter':
                img = cv2.GaussianBlur(img, (5, 5), 1.5)
            elif method == 'luminance':
                brightness_factor = random.random() * 0.8 + 0.6
                table = np.array([i * brightness_factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
                img = cv2.LUT(img, table)

        result = np.zeros(np.shape(img), dtype=np.float32)
        img = cv2.normalize(img, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        clip_temp[i, :, :, :] = img
    clip = clip_temp.transpose((3, 0, 1, 2))
    return clip


def main():
    vis = visdom.Visdom(env='sequence_tester')
    fold = 'div3'
    folder = 'folder3'
    model_type = '_90_'
    # model_type = '_100_'
    test_video_idx = 0
    print('test video idx:', test_video_idx)

    # Parameters
    # sequentiaol_model_name = 'transformer'
    sequentiaol_model_name = 'LSTM'

    class end2end(nn.Module):
        def __init__(self, model2, insight_length=100):
            super(end2end, self).__init__()
            self.insight_length = insight_length
            self.model1 = Endo3D()
            self.model2 = model2

        def forward(self, x, tar_seq, pred_tar=False):
            # batch_size = tar_seq.size(0)
            fc8s = self.model1.forward_cov(x[0].cuda())[1].unsqueeze(1)
            for i in range(self.insight_length - 1):
                fc8s = torch.cat((fc8s, self.model1.forward_cov(x[i + 1].cuda())[1].unsqueeze(1)), 1)
            if pred_tar:
                tar_seq = [torch.max(self.model1.forward_cov(x[i].cuda())[0].cpu().data, 1)[1].numpy()[0]
                           for i in range(100)]
                tar_seq = torch.tensor(tar_seq).long().unsqueeze(0)
                print('endo3D:', tar_seq)
            output_ = self.model2.forward(fc8s, tar_seq.cuda())
            return output_

    # class end2end(nn.Module):
    #     def __init__(self, model2):
    #         super(end2end, self).__init__()
    #         self.model1 = Endo3D()
    #         self.model2 = model2
    #
    #     def forward(self, x, tar_seq):
    #         batch_size = tar_seq.size(0)
    #         x1 = torch.zeros(batch_size, 100, 1200).cuda()
    #         for i in range(100):
    #             _, fc8 = self.model1.forward_cov(x[i].cuda())
    #             # fc8.retain_grad()
    #             x1[:, i, :] = fc8.clone()
    #         # x1.retain_grad()
    #         output_ = self.model2.forward(x1, tar_seq.cuda())
    #         return output_

    print('The sequential model chosen is: %s %s' % (sequentiaol_model_name, model_type))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if model_type == '_90_':
        if sequentiaol_model_name == 'transformer':
            sequential_model = Transformer(trg_pad_idx=6, n_trg_vocab=7,
                                           d_word_vec=1200, d_model=1200, d_inner=1000,
                                           n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=100)
            model = end2end(sequential_model)
            model.model1.load_state_dict(torch.load('./params/cross_validation/' + fold + '/params_endo3d.pkl', map_location='cuda:0'))
            model.model2.load_state_dict(torch.load('./params/cross_validation/' + fold + '/params_trans_90_seq.pkl', map_location='cuda:0'))
            # model.load_state_dict(
            #     torch.load('./params/cross_validation/' + fold + '/params_trans_90_end2end.pkl'))
        else:
            sequential_model = seq2seq_LSTM(hidden_dim=1200, num_layers=3)
            model = end2end(sequential_model)
            model.model1.load_state_dict(torch.load('./params/cross_validation/' + fold + '/params_endo3d.pkl', map_location='cuda:0'))
            model.model2.load_state_dict(torch.load('./params/cross_validation/' + fold + '/params_LSTM_90_seq.pkl', map_location='cuda:0'))
            # model.load_state_dict(
            #     torch.load('./params/cross_validation/' + fold + '/params_LSTM_90_end2end.pkl'))
    elif model_type == '_100_':
        if sequentiaol_model_name == 'transformer':
            sequential_model = Transformer(trg_pad_idx=6, n_trg_vocab=7,
                                           d_word_vec=1200, d_model=1200, d_inner=1000,
                                           n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=100)
            model = end2end(sequential_model)
            model.load_state_dict(
                torch.load('./params/cross_validation/' + fold + '/params_trans_100_end2end.pkl'))
        else:
            sequential_model = seq2seq_LSTM(hidden_dim=1200, num_layers=3)
            model = end2end(sequential_model)
            model.load_state_dict(
                torch.load('./params/cross_validation/' + fold + '/params_LSTM_100_end2end.pkl'))
    model = nn.DataParallel(model)
    model = model.cuda()
    print("Model initialized.")

    current_path = os.path.abspath(os.getcwd())
    videos_path = os.path.join(current_path, 'data/sacro_jpg')
    with open(os.path.join(videos_path, 'dataset_' + fold + '.json'), 'r') as json_data:
        temp = json.load(json_data)
    test_list = temp['test']

    open_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_sequence/whole'
                             '/sequence_in_clips', test_list[test_video_idx])

    file = open(os.path.join(open_path, 'clip_seq.pickle'), 'rb')
    clip_seq = pickle.load(file)
    file.close()

    file = open(os.path.join(open_path, 'clip_seq_labels.pickle'), 'rb')
    clip_seq_label = pickle.load(file)
    file.close()

    # make a list that contains all the index for all sequences
    index = list(range(len(clip_seq)))
    slwin_size = 100
    step = 10
    start_index = 0
    sequence_length = len(clip_seq)
    slinwin_list = []
    while start_index + slwin_size <= len(index):
        cur_slwin = index[start_index: start_index + slwin_size]
        slinwin_list.append(cur_slwin)
        start_index += step
    if start_index + slwin_size - step != len(index):
        cur_slwin = index[-slwin_size:]
        slinwin_list.append(cur_slwin)

    cm = np.zeros((7, 7))
    if model_type == '_90_':
        # Apply the sequential model 90
        # make an array for all predicted sequence
        sequece_label = torch.zeros(len(slinwin_list) + 1, sequence_length)
        sequece_label[:, :] = torch.tensor(float('nan'))
        sequece_label[0, 0:90] = torch.tensor(clip_seq_label[0:90])
        # sequece_label[0, :] = torch.tensor(np.array(seq_true).reshape(1, -1)[0, 0:90])
        for i, cur_slwin in enumerate(slinwin_list):
            trg_seq = sequece_label[i, cur_slwin[0]:cur_slwin[90]].unsqueeze(0).long()  # the current 90 clips
            # trg_seq = torch.tensor(clip_seq_label[cur_slwin[0]:cur_slwin[90]]).unsqueeze(0).long()
            sequence_input = [torch.tensor(np.float32(augmentation(clip_seq[idx]))).unsqueeze(0) for idx in cur_slwin]
            labels = torch.tensor(clip_seq_label[cur_slwin[10]:cur_slwin[-1] + 1])

            with torch.no_grad():
                sequence_output = model(sequence_input, trg_seq)
            _, predicted_labels = torch.max(sequence_output.cpu().data, 2)
            print(predicted_labels)
            # cm += confusion_matrix(labels.numpy().reshape(-1, ), predicted_labels.view(-1, ),
            #                        labels=[0, 1, 2, 3, 4, 5, 6])
            sequece_label[i + 1, cur_slwin[10]:(cur_slwin[-1] + 1)] = predicted_labels  # save the predictions to the next
        # sequece_label[0, :] = torch.tensor(float('nan'))
        sequece_label, _ = torch.mode(sequece_label, 0)
        # sequece_label= mode_average(sequece_label.numpy(), slinwin_sz=slin_sz)
        sequece_label = sequece_label.numpy()
    elif model_type == '_100_':
        # Apply the sequential model 100
        sequece_label = torch.zeros(len(slinwin_list), sequence_length)
        sequece_label[:, :] = torch.tensor(float('nan'))
        for i, cur_slwin in enumerate(slinwin_list):
            sequence_input = [torch.tensor(np.float32(augmentation(clip_seq[idx]))).unsqueeze(0) for idx in cur_slwin]
            labels = torch.tensor(clip_seq_label[cur_slwin[0]:cur_slwin[-1] + 1])
            trg_seq = 0
            with torch.no_grad():
                sequence_output = model(sequence_input, trg_seq, pred_tar=True)
            _, predicted_labels = torch.max(sequence_output.cpu().data, 2)
            print('seq_model:', predicted_labels)
            print('labels:', labels)
            cm += confusion_matrix(labels.numpy().reshape(-1, ), predicted_labels.view(-1, ),
                                   labels=[0, 1, 2, 3, 4, 5, 6])
            sequece_label[i, cur_slwin[0]:(cur_slwin[-1] + 1)] = predicted_labels
        sequece_label, _ = torch.mode(sequece_label, 0)
        # mode_av = mode_average(sequece_label.numpy(), slinwin_sz=slin_sz)
        sequece_label = sequece_label.numpy()
    cm = confusion_matrix(clip_seq_label, list(sequece_label), labels=[0, 1, 2, 3, 4, 5, 6])
    print(cm / np.sum(cm, axis=1)[:, None] * 100)

    save_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/results', folder, test_list[test_video_idx])

    file = open(os.path.join(save_path, 'seq_' + sequentiaol_model_name + model_type + 'end2end.pickle'), 'wb')
    pickle.dump(list(sequece_label), file)
    file.close()


if __name__ == '__main__':
    main()
