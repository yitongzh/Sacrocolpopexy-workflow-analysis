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
from cholec80_phase.model import Endo3D
import pickle


class cholec_sequence_loder(data.Dataset):
    def __init__(self, device, mode='train', datatype='current', train_epoch_size=1000, validation_epoch_size=600,
                 building_block=True, sliwin_sz=300):
        current_path = os.path.abspath(os.getcwd())
        self.videos_path = os.path.join('/home/yitong/venv_yitong/cholec80_phase/data/cholec_jpg')
        self.batch_num = train_epoch_size
        self.validation_batch_size = validation_epoch_size
        self.mode = mode
        self.datatype = datatype
        self.phases = ['1_Preparation', '2_CalotTriangleDissection', '3_ClippingCutting', '4_GallbladderDissection',
                       '5_GallbladderPackaging', '6_CleaningCoagulation', '7_GallbladderRetraction']
        self.aug_methods = ['non', 'random_flip', 'random_rot', 'crop', 'Gauss_filter', 'luminance']

        self.sliwin_sz = sliwin_sz

        self.device = device
        self.model = Endo3D().to(self.device)
        self.model.load_state_dict(torch.load('./params/params_c3d_1200.pkl'))
        self.model.eval()
        self.random_sample_counter = 0

        self.anchor_random_shift = 0.3

        self.train_video_list = ['video' + str(i).zfill(2) for i in range(1, 41)]
        self.validation_video_list = ['video' + str(i).zfill(2) for i in range(41, 61)]
        self.test_video_list = ['video' + str(i).zfill(2) for i in range(61, 81)]

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
        video_ls = []
        sort_ls = []
        for root, _, files in os.walk(video_path, topdown=False):
            for name in files:
                if name.endswith('.jpg'):
                    video_ls.append(os.path.join(root, name))
                    sort_ls.append(name)

        sort_idx = np.argsort([int(item.split('.')[0][2:]) for item in sort_ls])
        video_ls = [video_ls[idx] for idx in sort_idx]

        total_seq_len = len(video_ls)
        down_sample_ratio = int(total_seq_len / (16 * 1000))

        anchor = []
        last_phase = video_ls[0].split('/')[-2]
        last_item = video_ls[0]
        for item in video_ls:
            current_phase = item.split('/')[-2]
            if current_phase != last_phase:
                anchor.append(last_item)
            last_phase = current_phase
            last_item = item
        anchor.append(video_ls[-1])


        # shift the anchor frame to the middle of the sliding window
        anchor = [int(video_ls.index(item) - (self.sliwin_sz * 16 * down_sample_ratio -
                                              (down_sample_ratio - 1))/2) for item in anchor]
        end_frame = max(anchor)
        anchor.remove(max(anchor))

        # adjust the last anchor so the sequence will not exceed the video length
        total_frame_num = self.sliwin_sz * 16 * down_sample_ratio - (down_sample_ratio - 1)
        anchor.append(end_frame - total_frame_num - int(total_frame_num * 0.5 * self.anchor_random_shift))
        return video_ls, anchor

    # def get_sliding_window(self, video_ls):
    #     total_frame_num = self.sliwin_sz * 160 - 9
    #     total_seq_len = len(video_ls)
    #     max_frame_idx = total_seq_len - total_frame_num + self.non_cur_sz * 160
    #     min_frame_idx = 1 - self.non_cur_sz * 160
    #     start_img = random.randint(min_frame_idx, max_frame_idx)
    #     self.random_sample_counter += 1
    #     sliwin = []
    #     for i in range(self.sliwin_sz):
    #         st_img = start_img + i * 160
    #         sliwin.append([video_ls[st_img + j * 10] if 0 < st_img + j * 10 < len(video_ls) else 6 for j in range(16)])
    #     return sliwin

    def get_sliding_window_anchor(self, video_ls, anchor):
        total_seq_len = len(video_ls)
        # down_sample_ratio = int(total_seq_len / (16 * 1000))
        down_sample_ratio = 10

        total_frame_num = self.sliwin_sz * 16 * down_sample_ratio - (down_sample_ratio - 1)
        max_frame_idx = total_seq_len - total_frame_num
        min_frame_idx = 1

        rand_anchor = [item + int(total_frame_num * 0.5 * self.anchor_random_shift *
                                  random.uniform(-1, 1)) for item in anchor]
        start_img = random.choice(rand_anchor)
        if start_img < min_frame_idx or start_img > max_frame_idx:
            # print('start_img exceeds the bound, sample the sequence randomly')
            self.random_sample_counter += 1
            start_img = random.randint(min_frame_idx, max_frame_idx)
        sliwin = []
        for i in range(self.sliwin_sz):
            st_img = start_img + i * 16 * down_sample_ratio
            sliwin.append([video_ls[st_img + j * down_sample_ratio] for j in range(16)])
        return sliwin

    def get_sliding_window_whole(self, video_ls):
        total_seq_len = len(video_ls)
        # down_sample_ratio = int(total_seq_len / (16 * 1000))
        down_sample_ratio = 10

        self.sliwin_sz = int((len(video_ls) + (down_sample_ratio - 1)) / (16 * down_sample_ratio))
        start_img = 0
        sliwin = []
        for i in range(self.sliwin_sz):
            st_img = start_img + i * (16 * down_sample_ratio)
            sliwin.append([video_ls[st_img + j * down_sample_ratio] for j in range(16)])
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
        SW_input = torch.zeros(self.sliwin_sz, 1200)
        labels_cur = []
        labels_pred = []
        for i, clip_ls in enumerate(sliwin):
            label, clip = self.assemble_single_clip(clip_ls, augmentation_method, seed)
            clip = np.float32(clip.transpose((3, 0, 1, 2)))  # change to 3, 16, 112, 112
            with torch.no_grad():
                output, clip4200 = self.model.forward_cov(torch.from_numpy(clip).unsqueeze(0).to(self.device))
            # print(self.model)
            _, output = torch.max(output.cpu().data, 1)
            SW_input[i] = clip4200.cpu()
            labels_pred.append(output.item())
            labels_cur.append(label)

        return labels_cur, SW_input, labels_pred

    def whole_len_output(self, video_name):
        # vis = visdom.Visdom(env='paly_ground')
        video_path = os.path.join(self.videos_path, video_name)
        video_ls, anchor = self.get_video_ls(video_path)
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

                # Remove the normalization only with end2end model
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

    def build_epoch(self):
        print('building the training set...')
        self.mode = 'train'
        self.epoch_train_inputs = []
        self.epoch_train_labels_cur = []
        self.epoch_train_labels_pred = []
        for batch in tqdm(range(self.batch_num), ncols=80):
            labels, inputs, labels_pred = self.get_sequence_of_clips()
            self.epoch_train_inputs.append(inputs)
            self.epoch_train_labels_cur.append(labels)
            self.epoch_train_labels_pred.append(labels_pred)
        # print('Training set statistics:', Counter(self.epoch_train_labels))

    def build_validation(self):
        print('building the training set...')
        self.mode = 'validation'
        self.epoch_validation_inputs = []
        self.epoch_validation_labels_cur = []
        self.epoch_validation_labels_pred = []
        for batch in tqdm(range(self.validation_batch_size), ncols=80):
            labels, inputs, labels_pred = self.get_sequence_of_clips()
            self.epoch_validation_inputs.append(inputs)
            self.epoch_validation_labels_cur.append(labels)
            self.epoch_validation_labels_pred.append(labels_pred)
        # print('Training set statistics:', Counter(self.epoch_train_labels))


if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # cholec = cholec_sequence_loder(device, train_epoch_size=1000, validation_epoch_size=1000, sliwin_sz=100)
    # video_list = ['video' + str(i).zfill(2) for i in range(1, 81)]
    # for video_name in video_list:
    #     cholec.whole_len_output(video_name)
    # # for labels_val, inputs_val in whole_loder:
    # #     print(labels_val)

    folders = ['folder1', 'folder2', 'folder3', 'folder4', 'folder5']
    # sacro = sacro_loder(building_block=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mode = 'train'
    # mode = 'validation'
    sacro = cholec_sequence_loder(device, train_epoch_size=1000, validation_epoch_size=1000, sliwin_sz=100)
    if mode == 'train':
        for folder in folders:
            # folder = 'folder5'
            print('mode: %s save to: %s' % (mode, folder))
            sacro.build_epoch()

            train_input = open('./data/anchor/train/' + folder + '/train_input.pickle', 'wb')
            pickle.dump(sacro.epoch_train_inputs, train_input)
            train_input.close()

            label_curr = open('./data/anchor/train/' + folder + '/label_curr.pickle', 'wb')
            pickle.dump(sacro.epoch_train_labels_cur, label_curr)
            label_curr.close()
            print('Counter for current labels')
            print(Counter(np.array(sacro.epoch_train_labels_cur).reshape(-1, )))

            label_pred = open('./data/anchor/train/' + folder + '/label_pred.pickle', 'wb')
            pickle.dump(sacro.epoch_train_labels_pred, label_pred)
            label_pred.close()
            print('Counter for predicted labels')
            print(Counter(np.array(sacro.epoch_train_labels_pred).reshape(-1, )))

            print('train set saved, the total number of random re-sampling is %i' % sacro.random_sample_counter)
            sacro.random_sample_counter = 0
    else:
        print('mode:', mode)
        sacro.build_validation()

        validation_input = open('./data/anchor/validation/validation_input.pickle', 'wb')
        pickle.dump(sacro.epoch_validation_inputs, validation_input)
        validation_input.close()

        label_curr = open('./data/anchor/validation/label_curr.pickle', 'wb')
        pickle.dump(sacro.epoch_validation_labels_cur, label_curr)
        label_curr.close()
        print('Counter for current labels')
        print(Counter(np.array(sacro.epoch_validation_labels_cur).reshape(-1, )))

        label_pred = open('./data/anchor/validation/label_pred.pickle', 'wb')
        pickle.dump(sacro.epoch_validation_labels_pred, label_pred)
        label_pred.close()
        print('Counter for predicted labels')
        print(Counter(np.array(sacro.epoch_validation_labels_pred).reshape(-1, )))

        print('validation set saved, the total number of random re-sampling is %i' % sacro.random_sample_counter)
