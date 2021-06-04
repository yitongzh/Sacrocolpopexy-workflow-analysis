#! /usr/bin/env python
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


class cholec80_loder(data.Dataset):
    def __init__(self, mode='train', train_epoch_size=5000, validation_epoch_size=500, building_block=True):
        current_path = os.path.abspath(os.getcwd())
        self.videos_path = os.path.join('/home/yitong/venv_yitong/cholec80_phase/data/cholec_jpg')
        # self.batch_size = batch_size
        self.batch_num = train_epoch_size
        self.validation_batch_size = validation_epoch_size
        self.mode = mode
        self.phases = ['1_Preparation', '2_CalotTriangleDissection', '3_ClippingCutting', '4_GallbladderDissection',
                       '5_GallbladderPackaging', '6_CleaningCoagulation', '7_GallbladderRetraction']
        self.aug_methods = ['non', 'random_flip', 'random_rot', 'crop', 'Gauss_filter', 'luminance']
        self.aug_methods_prop = {'random_flip': 0.1, 'random_rot': 0.2,
                                 'crop': 0.3, 'Gauss_filter': 0.1, 'luminance': 0.2}

        self.train_video_list = ['video' + str(i).zfill(2) for i in range(1, 41)]
        self.validation_video_list = ['video' + str(i).zfill(2) for i in range(41, 61)]
        if building_block:
            self.build_epoch()
            self.build_validation()

    def __getitem__(self, idx):
        if self.mode == 'train':
            labels = self.epoch_train_labels[idx]
            inputs = self.epoch_train_inputs[idx]
        elif self.mode == 'validation':
            labels = self.validation_labels[idx]
            inputs = self.validation_inputs[idx]
        return labels, inputs

    def __len__(self):
        if self.mode == 'train':
            return self.batch_num
        elif self.mode == 'validation':
            return self.validation_batch_size

    def augmentation(self, img, method, seed):
        img = img[:, :, -1::-1]
        random.seed(seed)
        if method != 'non':
            if random.random()< self.aug_methods_prop['random_flip']:
                flip_num = random.choice([-1, 0, 1])
                img = cv2.flip(img, flip_num)
            if random.random() < self.aug_methods_prop['random_rot']:
                rows, cols, _ = np.shape(img)
                rot_angle = random.random() * 360
                # rot_angle = random.uniform(-1, 1) * 15
                M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), rot_angle, 1)
                img = cv2.warpAffine(img, M, (cols, rows))
            if random.random() < self.aug_methods_prop['crop']:
                rows, cols, _ = np.shape(img)
                result_size = random.randint(50, 150)
                result_row = random.randint(result_size, rows - result_size)
                result_col = random.randint(result_size, cols - result_size)
                img = img[result_row - result_size:result_row + result_size,
                      result_col - result_size:result_col + result_size, :]
            if random.random() < self.aug_methods_prop['Gauss_filter']:
                img = cv2.GaussianBlur(img, (5, 5), 1.5)
            if random.random() < self.aug_methods_prop['luminance']:
                brightness_factor = random.random() * 0.8 + 0.6
                table = np.array([i * brightness_factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
                img = cv2.LUT(img, table)

        img = cv2.resize(img, (112, 112))
        result = np.zeros(np.shape(img), dtype=np.float32)
        cv2.normalize(img, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return result

    def get_video_ls(self, video_path):
        video_ls = []
        sort_ls = []
        for root, _, files in os.walk(video_path, topdown=False):
            for name in files:
                if name.endswith('.jpg'):
                    video_ls.append(os.path.join(root, name))
                    sort_ls.append(name)

        sort_idx = np.argsort([int(item.split('.')[0][2:]) for item in sort_ls])
        video_ls = [video_ls[idx] for idx in sort_idx]
        return video_ls

    def define_clip(self, jpg_list, phase_path, down_sample_ratio):
        jpg_list.sort()

        item_kp = jpg_list[0]
        cut_point = [0]
        for i in range(1, len(jpg_list)):
            item_k = jpg_list[i]
            if item_k - item_kp != 1:
                cut_point.append(i)
            item_kp = item_k

        lists = []
        for i in range(len(cut_point)):
            if i != len(cut_point) - 1:
                A = jpg_list[cut_point[i]:cut_point[i + 1] - 1]
            else:
                A = jpg_list[cut_point[i]:]
            if len(A) > 15 * down_sample_ratio:
                lists.append(A)
        jpg_list = random.choice(lists)
        # if len(jpg_list) < 150:
        #     print(min(jpg_list), max(jpg_list))
        #     print(phase_path)
        start_img = random.randint(min(jpg_list), max(jpg_list) - 15 * down_sample_ratio)
        img_list = [('%s' % str(start_img + i * down_sample_ratio).zfill(8) + '.jpg') for i in range(16)]
        return img_list

    def get_clip(self):
        if self.mode == 'train':
            video_name = random.choice(self.train_video_list)
            # video_name = 'e8d3eabc-edd1-414f-9d95-277df742655a'
            augmentation_method = random.choice(self.aug_methods)
            # augmentation_method = self.aug_methods[0]
        elif self.mode == 'validation':
            video_name = random.choice(self.validation_video_list)
            augmentation_method = self.aug_methods[0]  # No augmentation for validation set

        video_ls = self.get_video_ls(os.path.join(self.videos_path, video_name))
        total_seq_len = len(video_ls)
        down_sample_ratio = int(total_seq_len / (16 * 1000))
        down_sample_ratio = 10

        while True:
            phase = random.choice(self.phases)
            jpg_list = [int(os.path.splitext(item)[0]) for item in
                        os.listdir(os.path.join(self.videos_path, video_name, phase))]
            if len(jpg_list) > 150:
                break

        # Get the number of frames in the selected phase
        # with open(os.path.join(self.videos_path, video_name, 'frames_in_phase.json'), 'r') as json_data:
        #     temp = json.load(json_data)
        # frame_num = temp[self.phases.index(phase)]
        img_list = self.define_clip(jpg_list, os.path.join(self.videos_path, video_name, phase), down_sample_ratio)

        seed = random.random()
        clip = np.zeros([16, 112, 112, 3])
        for i in range(len(img_list)):
            img = cv2.imread(os.path.join(self.videos_path, video_name, phase, img_list[i]))
            img = self.augmentation(img, augmentation_method, seed)
            clip[i, :, :, :] = img
        return self.phases.index(phase), clip

    def build_epoch(self):
        print('building the training set...')
        self.mode = 'train'
        self.epoch_train_inputs = []
        self.epoch_train_labels = []
        for batch in tqdm(range(self.batch_num), ncols=80):
            labels, inputs = self.get_clip()  # inputs 16, 112, 112, 3
            inputs = inputs.transpose((3, 0, 1, 2))  # change to 3, 16, 112, 112
            self.epoch_train_inputs.append(inputs)
            self.epoch_train_labels.append(labels)
        print('Training set statistics:', Counter(self.epoch_train_labels))

    def build_validation(self):
        print('building the validation set...')
        self.mode = 'validation'
        self.validation_inputs = []
        self.validation_labels = []
        for b in tqdm(range(self.validation_batch_size), ncols=80):
            labels, inputs = self.get_clip()  # inputs 16, 112, 112, 3
            inputs = inputs.transpose((3, 0, 1, 2))  # change to 3, 16, 112, 112
            self.validation_inputs.append(inputs)
            self.validation_labels.append(labels)
        print('Validation set statistics:', Counter(self.validation_labels))


if __name__ == "__main__":
    # sacro = sacro_loder(building_block=False)
    # sacro = cholec80_loder(train_epoch_size=5000, validation_epoch_size=600)
    # train_loader = data.DataLoader(sacro, 10, shuffle=True)
    # for epoch in range(10):
    #     for labels, inputs in tqdm(train_loader):
    #         print(labels.size())
    #     print(sacro.validation_inputs.size())
    #     if epoch == 5:
    #         sacro.build_epoch()

    # sacro.mode = 'validation'
    # _, clip = sacro.get_clip()
    # print(clip)
    # for i in range(len(sacro)):
    #     print(sacro[i][0].size(), sacro[i][1].size())

    # print(sacro.validation_labels.size(), sacro.validation_inputs.size())

    vis = visdom.Visdom(env='clip_viz')
    sacro = cholec80_loder(train_epoch_size=500, validation_epoch_size=10)
    train_loader = data.DataLoader(sacro, 5, shuffle=True)

    for labels, inputs in train_loader:
        # print(inputs[0, :, :, :, :].permute(1, 0, 2, 3))
        vis.images(inputs[0, :, :, :, :].permute(1, 0, 2, 3), win='clip_endo3D', opts=dict(title='endo3D'))
        time.sleep(1)
