from torch.utils import data
import os
import json
import sys
import random
import numpy as np
import torch
import pickle
import cv2

sys.path.append('../')


def clips_loder(video_name_list, clip_sequence_path):
    clips_seq_dict = {}
    for name in video_name_list:
        open_path = os.path.join(clip_sequence_path, name)

        a = open(os.path.join(open_path, 'clip_seq.pickle'), 'rb')
        clips_sequence = pickle.load(a)
        a.close()

        b = open(os.path.join(open_path, 'clip_seq_labels.pickle'), 'rb')
        clips_labels = pickle.load(b)
        b.close()
        pack = {'input': clips_sequence,
                'label': clips_labels}
        clips_seq_dict.update({name: pack})
    return clips_seq_dict


def sliwin_idx_generator(video_len, seq_len, seq_num):
    index = list(range(video_len))
    step_size = round(video_len / (seq_num + 1)) - 1
    start_index = 0
    slinwin_list = []
    while start_index + seq_len <= video_len:
        cur_slwin = index[start_index: start_index + seq_len]
        slinwin_list.append(cur_slwin)
        start_index += step_size
    while len(slinwin_list) != seq_num:
        idx = random.choice(range(len(slinwin_list) - 1))
        del slinwin_list[idx]
    untouched = slinwin_list[-1][-1] + 1 - video_len
    slinwin_list[-1] = [slinwin_list[-1][i] - int(untouched) for i in range(len(slinwin_list[-1]))]
    return slinwin_list


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
                rot_angle = random.uniform(-1, 1) * 90
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


class clip_sequence_loader(data.Dataset):
    def __init__(self, video_name_list, seq_num=200, seq_len=100, is_augmentation=True):
        self.seq_num = seq_num
        self.seq_len = seq_len
        self.video_name_list = video_name_list
        self.clip_sequence_path = '/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_sequence/whole' \
                                  '/sequence_in_clips'
        self.clips_seq_dict = clips_loder(self.video_name_list, self.clip_sequence_path)

        self.is_augmentation = is_augmentation
        self.aug_methods = ['non', 'random_flip', 'random_rot', 'crop', 'Gauss_filter', 'luminance']

        # print(self.clips_seq_dict[self.video_name_list[0]]['label'])
        self.sliwins = dict(zip(self.video_name_list,
                                [sliwin_idx_generator(len(self.clips_seq_dict[video_name]['label']),
                                                      self.seq_len, self.seq_num) for video_name in
                                 self.video_name_list]))
        # print(self.sliwins[self.video_name_list[0]])

    def __getitem__(self, idx):
        clips_sequence = []
        clips_sequence_labels = []
        if self.is_augmentation:
            methods = [random.choice(self.aug_methods) for i in range(len(self.video_name_list))]
            # methods = [self.aug_methods[4] for i in range(len(self.video_name_list))] # Just for test
        else:
            methods = [self.aug_methods[0] for i in range(len(self.video_name_list))]
        seeds = [random.random for i in range(len(self.video_name_list))]

        # iteration over the single sequence
        for i in range(self.seq_len):
            clips_pack_for_endo3D = np.zeros([len(self.video_name_list), 3, 16, 112, 112])
            labels_pack_for_endo3D = np.zeros([len(self.video_name_list), ])
            # iteration over the videos
            for j in range(len(self.video_name_list)):
                video_name = self.video_name_list[j]
                cur_sliwin = self.sliwins[video_name][idx]
                clip = self.clips_seq_dict[video_name]['input'][cur_sliwin[i]]
                label = self.clips_seq_dict[video_name]['label'][cur_sliwin[i]]

                clip = augmentation(clip, methods[j], seeds[j])
                clips_pack_for_endo3D[j, :, :, :, :] = clip
                labels_pack_for_endo3D[j] = label
            clips_sequence.append(torch.tensor(np.float32(clips_pack_for_endo3D)))
            clips_sequence_labels.append(labels_pack_for_endo3D)
        return {'inputs': clips_sequence, 'labels': torch.tensor(clips_sequence_labels).long().transpose(0, 1)}

    def __len__(self):
        return self.seq_num


if __name__ == "__main__":
    divs = ['div1', 'div2', 'div3', 'div4', 'div5', 'div6', 'div7']
    counter = 0
    current_path = os.path.abspath(os.getcwd())
    videos_path = os.path.join(current_path, 'data/sacro_jpg')
    with open(os.path.join(videos_path, 'dataset_' + divs[counter] + '.json'), 'r') as json_data:
        temp = json.load(json_data)

    video_list = temp['validation']
    # video_list = temp['train']
    data_loader = clip_sequence_loader(video_list)
    x = data_loader[100]
    # print(x['labels'].size())
