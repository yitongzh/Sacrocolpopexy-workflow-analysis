import json
import numpy as np
import os
import visdom
import cv2
import torch
import tqdm

vis = visdom.Visdom(env='clip_viz')
current_path = os.path.abspath(os.getcwd())
database = os.path.join(current_path, 'data/sacro_jpg')
with open(os.path.join(database, 'dataset_div7.json'), 'r') as json_data:
    temp = json.load(json_data)

train_video_list = temp['train'] + temp['validation'] + temp['test']
current_video = 13
print(current_video)
images_path = os.path.join(database, train_video_list[current_video], 'transition_phase')
image_names = [int(os.path.splitext(item)[0]) for item in os.listdir(images_path)]
image_names.sort()
image_names = [str(i)+'.jpg' for i in image_names]
print(len(image_names))
for i in range(0, len(image_names), 10):
    img = cv2.imread(os.path.join(images_path, image_names[i])).transpose(2, 0, 1)
    img = img[-1::-1, :, :]
    vis.image(img,  win='clip_endo3D',  opts=dict(title='endo3D'))
    vis.close('cur_best')
    vis.text(image_names[i], win='cur_best', opts=dict(title='Current Best'))

