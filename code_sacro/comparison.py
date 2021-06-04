import sys
from model import Endo3D
from many2many_LSTM import many2many_LSTM
from transformer.transformer import Transformer
from utils.sequence_loder import sequence_loder
import numpy as np
import visdom
from tqdm import tqdm

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
from utils.clip_sequence_loder import clip_sequence_loader
import cv2

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


def augmentation(clip, method, seed):
    clip_temp = clip.transpose((1, 2, 3, 0)).copy()
    random.seed(seed)
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


class end2end(nn.Module):
    def __init__(self, model2, insight_length=100):
        super(end2end, self).__init__()
        self.insight_length = insight_length
        self.model1 = Endo3D()
        self.model2 = model2

    def forward(self, x, device):
        batch_size = x[0].size(0)
        input_ = torch.zeros(batch_size, self.insight_length, 3, 16, 112, 112)
        for i in range(self.insight_length):
            input_[:, i, :, :, :, :] = x[i]
        input_ = input_.view(batch_size * self.insight_length, 3, 16, 112, 112)

        fc8s = self.model1.forward_cov(input_.to(device))[1]
        fc8s = fc8s.view(batch_size, self.insight_length, 1200)
        # for i in range(self.insight_length - 1):
        #     fc8s = torch.cat((fc8s, self.model1.forward_cov(x[i + 1].cuda())[1].unsqueeze(1)), 1)
        output_ = self.model2.forward(fc8s)
        return output_


def main():
    vis = visdom.Visdom(env='comparison')
    device_s = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    device_e = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    extensions = ['stan', 'cont', 'pred', 'noised', 'seq']

    # Parameters
    # sequentiaol_model_name = 'transformer'
    sequentiaol_model_name = 'LSTM'
    folder = 'div3'
    KF_Folder = 'folder3'
    # test_video = 1  # two videos in test set, write 0 or 1

    model_e = end2end(many2many_LSTM(hidden_dim=1200, num_layers=3)).to(device_e)
    model_e.model1.load_state_dict(torch.load('./params/cross_validation/' + folder + '/params_endo3d.pkl',
                                            map_location='cuda:0'))
    model_e.model2.load_state_dict(torch.load('./params/cross_validation/' + folder + '/params_LSTM_m2m.pkl',
                                            map_location='cuda:0'))
    # model_e.load_state_dict(
    #     torch.load('./params/cross_validation/' + folder + '/params_LSTM_m2m_e2e.pkl'))

    for test_video in list(range(2)):
        print('The sequential model chosen is: %s' % sequentiaol_model_name, '\n'
              'The cross validation folder is: %s' % KF_Folder, '\n'
              'The test video is: %d' % test_video)

        model = Endo3D().to(device_s)
        model.load_state_dict(torch.load('./params/cross_validation/' + folder + '/params_endo3d.pkl'))

        if sequentiaol_model_name == 'transformer':
            sequential_model = Transformer(trg_pad_idx=6, n_trg_vocab=7,
                                           d_word_vec=1200, d_model=1200, d_inner=1000,
                                           n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=100).to(device_s)
            sequential_model.load_state_dict(torch.load('./params/cross_validation/' + folder + '/params_trans_90_' +
                                                        extension + '.pkl'))
            # sequential_model.load_state_dict(torch.load('./params/params_trans_save_point.pkl'))
            slin_sz = 30
        else:
            sequential_model = many2many_LSTM(hidden_dim=1200, num_layers=3).to(device_s)
            sequential_model.load_state_dict(torch.load('./params/cross_validation/' + folder + '/params_LSTM_m2m.pkl'))
            # sequential_model.load_state_dict(torch.load('./params/params_LSTM_save_point.pkl'))
            slin_sz = 17

        current_path = os.path.abspath(os.getcwd())
        videos_path = os.path.join(current_path, 'data/sacro_jpg')
        with open(os.path.join(videos_path, 'dataset_' + folder + '.json'), 'r') as json_data:
            temp = json.load(json_data)
        test_video_list = temp['test']
        validation_video_list = temp['validation']

        video_name = test_video_list[test_video]
        open_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_sequence/whole',
                                 KF_Folder, video_name)

        sacro_validation = clip_sequence_loader([video_name], is_augmentation=False)
        clips_seq_dict = sacro_validation.clips_seq_dict[video_name]
        clip_sequence = clips_seq_dict['input']
        clip_sequence_whole = [augmentation(clip, 'non', 0) for clip in clip_sequence].copy()
        seq_true = clips_seq_dict['label']

        # make a list that contains all the index for all sequences
        index = list(range(len(clip_sequence)))
        slwin_size = 100
        step = 10
        start_index = 0
        sequence_length = len(clip_sequence)
        slinwin_list = []
        while start_index + slwin_size <= len(index):
            cur_slwin = index[start_index: start_index + slwin_size]
            slinwin_list.append(cur_slwin)
            start_index += step
        if start_index + slwin_size - step != len(index):
            cur_slwin = index[-slwin_size:]
            slinwin_list.append(cur_slwin)

        # Apply the sequential model 90
        # make an array for all predicted sequence
        sequece_label = torch.zeros(len(slinwin_list) + 1, sequence_length)
        sequece_label[:, :] = torch.tensor(float('nan'))
        # sequece_label[0, 0:90] = torch.tensor(mode_av.reshape(1, -1)[0, 0:90])
        sequece_label[0, 0:90] = torch.tensor(np.array(seq_true).reshape(1, -1)[0, 0:90])
        for i, cur_slwin in enumerate(slinwin_list):
            sequence_input = clip_sequence_whole[cur_slwin[0]: cur_slwin[-1]+1]
            sequence_input = [torch.tensor(np.float32(item)).unsqueeze(0) for item in sequence_input]

            test_sequence = clip_sequence[cur_slwin[0]: cur_slwin[-1]+1]
            test_sequence = [torch.tensor(np.float32(augmentation(clip, 'non', 0))).unsqueeze(0)
                             for clip in test_sequence]

            #print(sum([torch.norm(test_sequence[i] - sequence_input[i]) for i in range(100)]) / 100)

            with torch.no_grad():
                sequence_output = model_e(sequence_input, device_e)
            # sequence_input = torch.tensor(np.float32(sequence_input))
            # with torch.no_grad():
            #     output, x_fc = model.forward_cov(sequence_input.to(device_s))
            # fc8s = x_fc.view(1, 100, 1200)
            # with torch.no_grad():
            #     sequence_output = sequential_model.forward(fc8s)

            _, predicted_labels = torch.max(sequence_output.cpu().data, 2)
            sequece_label[i + 1, cur_slwin[10]:(cur_slwin[-1] + 1)] = predicted_labels[:, 10:]  # save the predictions to the next
            # print(predicted_labels)
            # row
        # sequece_label[0, :] = torch.tensor(float('nan'))
        sequece_label, _ = torch.mode(sequece_label, 0)
        # sequece_label= mode_average(sequece_label.numpy(), slinwin_sz=slin_sz)
        sequece_label = sequece_label.numpy()

        # vis.line(X=np.array(range(len(seq_pre))), Y=np.column_stack((sequece_label, np.array(seq_true))),
        #          win='Surgical workflow sequential', update='append',
        #          opts=dict(title='Surgical workflow sequential', showlegend=True, legend=['Prediction', 'Ground Truth']))

        cm = confusion_matrix(np.array(seq_true), sequece_label, labels=[0, 1, 2, 3, 4, 5, 6])
        cm = cm / np.sum(cm, axis=1) * 100
        cm_inner = cm[1:6, 1:6]
        cm_inner = cm_inner / np.sum(cm_inner, axis=1)
        anot = np.sum(np.diag(cm_inner)[1:]) / 5
        x = list(np.diag(cm_inner)[1:6])
        txt3 = ''.join(['Phase %d accuracy:%d%%   ' % (i + 1, x[i]) for i in range(len(x))])
        vis.text(txt3, win='cur_best', opts=dict(title='Current Best'))
        print('[validation accuracy: %.3f %% accuracy without transition phase: %.3f%%'
              % (np.sum(np.diag(cm)) / 6, anot))
        # vis.heatmap(X=cm, win='heatmap3', opts=dict(title='confusion matrix sequential',
        #                                             rownames=['t0', 't1', 't2', 't3', 't4', 't5'],
        #                                             columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5']))
        vis.line(X=np.array(range(len(seq_true))), Y=np.column_stack((sequece_label, np.array(seq_true))),
                 win='Surgical workflow sequential', update='append',
                 opts=dict(title='Surgical workflow sequential', showlegend=True, legend=['Prediction', 'Ground Truth']))

        vis.heatmap(X=cm, win='heatmap', opts=dict(title='confusion matrix',
                                                   rownames=['t0', 't1', 't2', 't3', 't4', 't5', 't_not'],
                                                   columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p_not']))

        save_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/results', KF_Folder, video_name)
        if not os.path.exists(os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/results', KF_Folder)):
            os.mkdir(os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/results', KF_Folder))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # file = open(os.path.join(save_path, 'seq_true.pickle'), 'wb')
        # pickle.dump(list(seq_true), file)
        # file.close()
        #
        # file = open(os.path.join(save_path, 'seq_pred.pickle'), 'wb')
        # pickle.dump(list(seq_pre), file)
        # file.close()
        #
        # file = open(os.path.join(save_path, 'seq_mode_av.pickle'), 'wb')
        # pickle.dump(list(mode_av), file)
        # file.close()

        # file = open(os.path.join(save_path, 'seq_' + sequentiaol_model_name + '_90_m2m_e2e.pickle'), 'wb')
        # pickle.dump(list(sequece_label), file)
        # file.close()


if __name__ == '__main__':
    main()