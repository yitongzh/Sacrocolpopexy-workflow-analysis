import sys
sys.path.append('../')
import torch
import visdom
from sacro_loder import sacro_loder
from clip_sequence_loder import clip_sequence_loader
from torch.utils import data
import json
import os
import time


vis = visdom.Visdom(env='clip_viz')
sacro = sacro_loder(train_epoch_size=10, validation_epoch_size=10, cv_div='dataset_div3.json')
train_loader = data.DataLoader(sacro, 5, shuffle=True)

videos_path = '/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_jpg'
with open(os.path.join(videos_path, 'dataset_div3.json'), 'r') as json_data:
    temp = json.load(json_data)
train_list = temp['train']
validation_list = temp['validation']

sacro_e2e = clip_sequence_loader(validation_list[0:2], is_augmentation=False)
x = sacro_e2e[120]['inputs']

# input_ = torch.zeros(2, 100, 3, 16, 112, 112)
# for i in range(100):
#     input_[:, i, :, :, :, :] = x[i]
# inputs = input_.view(2*100, 3, 16, 112, 112)
# inputs = input_.view(2, 100, 3, 16, 112, 112)
# print(torch.norm((inputs-input_)))


vis.images(x[0][0, :, :, :, :].permute(1, 0, 2, 3),  win='clip_e2e',  opts=dict(title='e2e'))


for labels, inputs in train_loader:
    # print(inputs[0, :, :, :, :].permute(1, 0, 2, 3))
    vis.images(inputs[0, :, :, :, :].permute(1, 0, 2, 3),  win='clip_endo3D',  opts=dict(title='endo3D'))
    break
