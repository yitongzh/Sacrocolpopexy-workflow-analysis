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


class sacro_loder(data.Dataset):
    def __init__(self, mode='train', train_epoch_size=5000, validation_epoch_size=500, building_block=True,
                 cv_div='dataset_div.json'):
        current_path = os.path.abspath(os.getcwd())
        self.videos_path = os.path.join(current_path, 'data/sacro_jpg')
        # self.batch_size = batch_size
        self.batch_num = train_epoch_size
        self.validation_batch_size = validation_epoch_size
        self.mode = mode
        self.phases = ['transition_phase', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'non_phase']
        self.aug_methods = ['non', 'random_flip', 'random_rot', 'crop', 'Gauss_filter', 'luminance']
        self.dataset = cv_div

        with open(os.path.join(self.videos_path, self.dataset), 'r') as json_data:
            temp = json.load(json_data)

        self.train_video_list = temp['train']
        self.validation_video_list = temp['validation']
        if building_block:
            # start = time.time()
            self.build_epoch()
            # elapsed = (time.time() - start)
            # print("Data loded, time used:", elapsed)
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
                brightness_factor = random.random()* 0.8 + 0.6
                table = np.array([i * brightness_factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
                img = cv2.LUT(img, table)

        img = cv2.resize(img, (112, 112))
        result = np.zeros(np.shape(img), dtype=np.float32)
        cv2.normalize(img, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return result

    def define_clip(self, jpg_list):
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
            if len(A) > 150:
                lists.append(A)
        jpg_list = random.choice(lists)
        start_img = random.randint(min(jpg_list), max(jpg_list) - 150)
        img_list = [('%s' % str(start_img + i * 10).zfill(8) + '.jpg') for i in range(16)]
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

        while True:
            phase = random.choice(self.phases)
            if video_name != 'e8d3eabc-edd1-414f-9d95-277df742655a':
                if phase != 'non_phase':
                    break
            else:
                if phase != 'non_phase' and phase != 'phase5':
                    break

        # Get the number of frames in the selected phase
        with open(os.path.join(self.videos_path, video_name, 'frames_in_phase.json'), 'r') as json_data:
            temp = json.load(json_data)
        frame_num = temp[self.phases.index(phase)]

        jpg_list = [int(os.path.splitext(item)[0]) for item in
                    os.listdir(os.path.join(self.videos_path, video_name, phase))]
        img_list = self.define_clip(jpg_list)

        seed = random.random()
        clip = np.zeros([16, 112, 112, 3])
        for i in range(len(img_list)):
            img = cv2.imread(os.path.join(self.videos_path, video_name, phase, img_list[i]))
            img = self.augmentation(img, augmentation_method, seed)
            clip[i, :, :, :] = img
        return self.phases.index(phase), clip

    # def build_epoch(self):
    #     print('building the training set...')
    #     self.mode = 'train'
    #     self.epoch_train_inputs = []
    #     self.epoch_train_labels = []
    #     for batch in range(self.batch_num):
    #         inputs = torch.Tensor(self.batch_size, 16, 112, 112, 3)
    #         labels = torch.Tensor(self.batch_size, 1)
    #         for b in range(self.batch_size):
    #             label, clip = self.get_clip()
    #             inputs[b, :, :, :, :] = torch.FloatTensor(clip)
    #             labels[b, :] = label
    #         inputs = inputs.permute(0, 4, 1, 2, 3)
    #         self.epoch_train_inputs.append(inputs)
    #         self.epoch_train_labels.append(labels)

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
    sacro = sacro_loder(batch_num=30, validation_batch_size=10)
    train_loader = data.DataLoader(sacro, 10, shuffle=True)
    for epoch in range(10):
        for labels, inputs in tqdm(train_loader):
            print(labels.size())
        print(sacro.validation_inputs.size())
        if epoch == 5:
            sacro.build_epoch()


    # sacro.mode = 'validation'
    # _, clip = sacro.get_clip()
    # print(clip)
    # for i in range(len(sacro)):
    #     print(sacro[i][0].size(), sacro[i][1].size())

    # print(sacro.validation_labels.size(), sacro.validation_inputs.size())
