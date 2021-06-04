#! /usr/bin/env python
import sys

sys.path.append('../')
import json
import numpy as np
import os
from torch.utils import data
import cv2
import random
import torch
import time
from tqdm import tqdm
from collections import Counter
import visdom
# from sacro_wf_analysis.model import Endo3D_for_sequence
from sacro_wf_analysis.model import Endo3D
import pickle


class sequence_loder(data.Dataset):
    def __init__(self, div, device, mode='train', datatype='current', train_epoch_size=1000, validation_epoch_size=600,
                 building_block=True, sliwin_sz=300):
        current_path = os.path.abspath(os.getcwd())
        self.videos_path = os.path.join(current_path, 'data/sacro_jpg')
        self.batch_num = train_epoch_size
        self.validation_batch_size = validation_epoch_size
        self.mode = mode
        self.datatype = datatype
        self.phases = ['transition_phase', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'non_phase']
        self.aug_methods = ['non', 'random_flip', 'random_rot', 'crop', 'Gauss_filter', 'luminance']

        self.non_cur_sz = int(sliwin_sz / 2)
        self.cur_sz = sliwin_sz
        self.sliwin_sz = sliwin_sz + 2 * self.non_cur_sz

        self.device = device
        self.model = Endo3D().to(self.device)
        self.div = div
        self.model.load_state_dict(torch.load('./params/cross_validation/' + self.div + '/params_endo3d.pkl'))
        self.model.eval()
        self.random_sample_counter = 0

        self.anchor_random_shift = 0.3

        with open(os.path.join(self.videos_path, 'dataset_' + self.div + '.json'), 'r') as json_data:
            temp = json.load(json_data)

        self.train_video_list = temp['train']
        self.validation_video_list = temp['validation']

    def __getitem__(self, idx):
        if self.mode == 'train':
            if self.datatype == 'past':
                labels = self.epoch_train_labels_past[idx]
            elif self.datatype == 'current':
                labels = self.epoch_train_labels_cur[idx]
            elif self.datatype == 'future':
                labels = self.epoch_train_labels_future[idx]
            inputs = self.epoch_train_inputs[idx]
        elif self.mode == 'validation':
            if self.datatype == 'past':
                labels = self.epoch_validation_labels_past[idx]
            elif self.datatype == 'current':
                labels = self.epoch_validation_labels_cur[idx]
            elif self.datatype == 'future':
                labels = self.epoch_validation_labels_future[idx]
            inputs = self.epoch_validation_inputs[idx]
        elif self.mode == 'whole':
            labels = self.whole_labels[idx]
            inputs = self.whole_inputs[idx]
        return labels, inputs

    def __len__(self):
        if self.mode == 'train':
            return self.batch_num
        elif self.mode == 'validation':
            return self.validation_batch_size
        elif self.mode == 'whole':
            return len(self.whole_labels)

    def get_video_ls(self, video_path):
        def filtering_non_phase(video_ls_item):
            if video_ls_item.split('/')[-1].split('.')[0][0] == '0':
                return False
            else:
                return True

        video_ls = []
        sort_ls = []
        for root, _, files in os.walk(video_path, topdown=False):
            for name in files:
                if name.endswith('.jpg'):
                    video_ls.append(os.path.join(root, name))
                    sort_ls.append(name)

        sort_idx = np.argsort([int(item.split('.')[0][2:]) for item in sort_ls])
        video_ls = [video_ls[idx] for idx in sort_idx]
        video_ls = [item for item in filter(filtering_non_phase, video_ls)]

        with open(os.path.join(video_path, 'frame_pos_dict.json'), 'r') as json_data:
            temp = json.load(json_data)
        # find the last frame in a single phase, exclude the non-phase frame and transition phase frames
        anchor = [os.path.join(video_path, 'phase' + k[0], k)
                  for (k, v) in temp.items() if v == 100.0 and k[0] != '0' and k[0] != '6']

        # shift the anchor frame to the middle of the sliding window
        anchor = [int(video_ls.index(item) - (self.sliwin_sz * 160 - 9)/2) for item in anchor]
        end_frame = max(anchor)
        anchor.remove(max(anchor))

        # adjust the last anchor so the sequence will not exceed the video length
        total_frame_num = self.sliwin_sz * 160 - 9
        anchor.append(end_frame - total_frame_num + self.non_cur_sz * 160 -
                      int(total_frame_num * 0.5 * self.anchor_random_shift))
        return video_ls, anchor

    def get_sliding_window_whole(self, video_ls):
        total_frame_num = self.sliwin_sz * 160 - 9
        total_seq_len = len(video_ls)
        max_frame_idx = total_seq_len - total_frame_num
        start_img = 0
        sliwin = []
        for i in range(self.sliwin_sz):
            st_img = start_img + i * 160
            sliwin.append([video_ls[st_img + j * 10] for j in range(16)])
        return sliwin

    def augmentation(self, img, method, seed):
        img = img[:, :, -1::-1]
        random.seed(seed)
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
                result_size = random.randint(50, 150)
                result_row = random.randint(result_size, rows - result_size)
                result_col = random.randint(result_size, cols - result_size)
                img = img[result_row - result_size:result_row + result_size,
                      result_col - result_size:result_col + result_size, :]
            # elif method == 'Lap_filter':
            #     img = cv2.GaussianBlur(img, (5, 5), 1.5)
            #     img = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
            #     img = cv2.convertScaleAbs(img)
            elif method == 'Gauss_filter':
                img = cv2.GaussianBlur(img, (5, 5), 1.5)
            elif method == 'luminance':
                brightness_factor = random.random() * 0.8 + 0.6
                table = np.array([i * brightness_factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
                img = cv2.LUT(img, table)

        img = cv2.resize(img, (112, 112))
        result = np.zeros(np.shape(img), dtype=np.float32)
        img = cv2.normalize(img, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return img

    def assemble_single_clip(self, clip_ls, augmentation_method, seed):
        clip = np.zeros([16, 112, 112, 3])
        for i in range(len(clip_ls)):
            img = cv2.imread(clip_ls[i])
            img = self.augmentation(img, augmentation_method, seed)
            clip[i, :, :, :] = img
        nums = [self.phases.index(clip_ls[i].split('/')[-2]) for i in range(len(clip_ls))]
        label = sorted(nums)[len(nums) // 2]
        return label, clip

    def get_sequence_of_clips(self):
        # vis = visdom.Visdom(env='paly_ground')
        if self.mode == 'train':
            video_name = random.choice(self.train_video_list)
            augmentation_method = random.choice(self.aug_methods)
            # augmentation_method = self.aug_methods[0]
        elif self.mode == 'validation':
            video_name = random.choice(self.validation_video_list)
            # video_name = 'e8d3eabc-edd1-414f-9d95-277df742655a'
            augmentation_method = self.aug_methods[0]  # No augmentation for validation set

        video_path = os.path.join(self.videos_path, video_name)
        video_ls, anchor = self.get_video_ls(video_path)
        sliwin = self.get_sliding_window_anchor(video_ls, anchor)
        # sliwin = self.get_sliding_window(video_ls)

        seed = random.random()
        SW_input = torch.zeros(self.cur_sz, 1200)
        labels_cur = []
        labels_past = []
        labels_future = []
        labels_pred = []
        for i, clip_ls in enumerate(sliwin):
            if i < self.non_cur_sz:
                if clip_ls == 6:
                    label = 6
                else:
                    nums = [self.phases.index(item.split('/')[-2]) if item != 6 else 6 for item in clip_ls]
                    label = sorted(nums)[len(nums) // 2]
                labels_past.append(label)
            elif self.non_cur_sz <= i < self.non_cur_sz + self.cur_sz:
                label, clip = self.assemble_single_clip(clip_ls, augmentation_method, seed)
                # vis.image(clip[-1, :, :, :].squeeze().transpose(2, 1, 0) * 255, win='video~')
                # vis.text(label, win='label')
                clip = np.float32(clip.transpose((3, 0, 1, 2)))  # change to 3, 16, 112, 112
                output, clip4200 = self.model.forward_cov(torch.from_numpy(clip).unsqueeze(0).to(self.device))
                # print(self.model)
                _, output = torch.max(output.cpu().data, 1)
                SW_input[i - self.non_cur_sz * 2] = clip4200.cpu()
                labels_pred.append(output.item())
                labels_cur.append(label)
            elif i >= self.non_cur_sz + self.cur_sz:
                if clip_ls == 6:
                    label = 6
                else:
                    nums = [self.phases.index(item.split('/')[-2]) if item != 6 else 6 for item in clip_ls]
                    label = sorted(nums)[len(nums) // 2]
                labels_future.append(label)

        # print(len(labels_past))
        # print(labels_past)
        # print(len(labels_cur))
        # print(labels_cur)
        # print(len(labels_future))
        # print(labels_future)
        return [labels_past, labels_cur, labels_future], SW_input, labels_pred

    def whole_len_output(self, video_name):
        # vis = visdom.Visdom(env='paly_ground')
        video_path = os.path.join(self.videos_path, video_name)
        video_ls, anchor = self.get_video_ls(video_path)
        self.sliwin_sz = int((len(video_ls) + 9) / 160)
        sliwin = self.get_sliding_window_whole(video_ls)

        self.whole_inputs = []
        self.whole_labels = []
        for clip_ls in tqdm(sliwin, ncols=80):
            # for idx, clip_ls in tqdm(enumerate(sliwin), ncols=80):
            clip = np.zeros([16, 112, 112, 3])
            for i in range(len(clip_ls)):
                img = cv2.imread(clip_ls[i])
                img = img[:, :, -1::-1]
                img = cv2.resize(img, (112, 112))
                result = np.zeros(np.shape(img), dtype=np.float32)
                img = cv2.normalize(img, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                clip[i, :, :, :] = img
            nums = [self.phases.index(clip_ls[i].split('/')[-2]) for i in range(len(clip_ls))]
            label = sorted(nums)[len(nums) // 2]
            # vis.image(img.transpose(2, 0, 1) * 255, win='video~')
            # vis.text(label, win='label')
            inputs = clip.transpose((3, 0, 1, 2))  # change to 3, 16, 112, 112
            self.whole_inputs.append(inputs)
            self.whole_labels.append(label)
        self.mode = 'whole'


if __name__ == "__main__":
    folders = ['folder1', 'folder2', 'folder3', 'folder4', 'folder5']
    # sacro = sacro_loder(building_block=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    div = 'div7'
    mode = 'train'
    mode = 'validation'
    sacro = sequence_loder(div, device, train_epoch_size=1000, validation_epoch_size=1000, sliwin_sz=100)
    if mode == 'train':
        for folder in folders:
            # folder = 'folder5'
            print('mode: %s save to: %s, %s' % (mode, div, folder))
            sacro.build_epoch()

            train_input = open('./data/sacro_sequence/train/' + div + '/' + folder + '/train_input.pickle', 'wb')
            pickle.dump(sacro.epoch_train_inputs, train_input)
            train_input.close()

            label_past = open('./data/sacro_sequence/train/' + div + '/' + folder + '/label_past.pickle', 'wb')
            pickle.dump(sacro.epoch_train_labels_past, label_past)
            label_past.close()
            print('Counter for past labels')
            print(Counter(np.array(sacro.epoch_train_labels_past).reshape(-1, )))

            label_curr = open('./data/sacro_sequence/train/' + div + '/' + folder + '/label_curr.pickle', 'wb')
            pickle.dump(sacro.epoch_train_labels_cur, label_curr)
            label_curr.close()
            print('Counter for current labels')
            print(Counter(np.array(sacro.epoch_train_labels_cur).reshape(-1, )))

            label_future = open('./data/sacro_sequence/train/' + div + '/' + folder + '/label_future.pickle', 'wb')
            pickle.dump(sacro.epoch_train_labels_future, label_future)
            label_future.close()
            print('Counter for future labels')
            print(Counter(np.array(sacro.epoch_train_labels_future).reshape(-1, )))

            label_pred = open('./data/sacro_sequence/train/' + div + '/' + folder + '/label_pred.pickle', 'wb')
            pickle.dump(sacro.epoch_train_labels_pred, label_pred)
            label_pred.close()
            print('Counter for predicted labels')
            print(Counter(np.array(sacro.epoch_train_labels_pred).reshape(-1, )))

            print('train set saved, the total number of random re-sampling is %i' % sacro.random_sample_counter)
            sacro.random_sample_counter = 0
    else:
        folder = 'folder7'
        print('mode: %s save to: %s' % (mode, folder))
        sacro.build_validation()

        validation_input = open('./data/sacro_sequence/validation/' + folder + '/validation_input.pickle', 'wb')
        pickle.dump(sacro.epoch_validation_inputs, validation_input)
        validation_input.close()

        label_past = open('./data/sacro_sequence/validation/' + folder + '/label_past.pickle', 'wb')
        pickle.dump(sacro.epoch_validation_labels_past, label_past)
        label_past.close()
        print('Counter for past labels')
        print(Counter(np.array(sacro.epoch_validation_labels_past).reshape(-1, )))

        label_curr = open('./data/sacro_sequence/validation/' + folder + '/label_curr.pickle', 'wb')
        pickle.dump(sacro.epoch_validation_labels_cur, label_curr)
        label_curr.close()
        print('Counter for current labels')
        print(Counter(np.array(sacro.epoch_validation_labels_cur).reshape(-1, )))

        label_future = open('./data/sacro_sequence/validation/' + folder + '/label_future.pickle', 'wb')
        pickle.dump(sacro.epoch_validation_labels_future, label_future)
        label_future.close()
        print('Counter for future labels')
        print(Counter(np.array(sacro.epoch_validation_labels_future).reshape(-1, )))

        label_pred = open('./data/sacro_sequence/validation/' + folder + '/label_pred.pickle', 'wb')
        pickle.dump(sacro.epoch_validation_labels_pred, label_pred)
        label_pred.close()
        print('Counter for predicted labels')
        print(Counter(np.array(sacro.epoch_validation_labels_pred).reshape(-1, )))

        print('validation set saved, the total number of random re-sampling is %i' % sacro.random_sample_counter)
