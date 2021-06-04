import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Flatten, Dropout, ZeroPadding3D
from model import Endo3D
import torch
import deepdish as dd
import os
from C3D_model import C3D


# model = Endo3D()
# Endo3D_state_dict = model.state_dict()
# pre_state_dict = torch.load('./params/c3d.pickle')
# print(pre_state_dict.keys())
# new_state_dict = {k: v for k, v in pre_state_dict.items() if k in Endo3D_state_dict}
# Endo3D_state_dict.update(new_state_dict)
# model.load_state_dict(Endo3D_state_dict)

# net = C3D()
# py_model = Endo3D()
# Endo3D_state_dict = py_model.state_dict()
#
# net.load_state_dict(torch.load('params/c3d.pickle'))
# pre_state_dict = net.state_dict()
#
# new_state_dict = {k: v for k, v in pre_state_dict.items() if k in Endo3D_state_dict}
# py_model.load_state_dict(new_state_dict)
# Endo3D_state_dict.update(new_state_dict)
# model.load_state_dict(Endo3D_state_dict)


def C3Dnet(nb_classes, input_shape):
    input_tensor = Input(shape=input_shape)
    # 1st block
    x = Conv3D(64, [3, 3, 3], activation='relu', padding='same', strides=(1, 1, 1), name='conv1')(input_tensor)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)
    # 2nd block
    x = Conv3D(128, [3, 3, 3], activation='relu', padding='same', strides=(1, 1, 1), name='conv2')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
    # 3rd block
    x = Conv3D(256, [3, 3, 3], activation='relu', padding='same', strides=(1, 1, 1), name='conv3a')(x)
    x = Conv3D(256, [3, 3, 3], activation='relu', padding='same', strides=(1, 1, 1), name='conv3b')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)
    # 4th block
    x = Conv3D(512, [3, 3, 3], activation='relu', padding='same', strides=(1, 1, 1), name='conv4a')(x)
    x = Conv3D(512, [3, 3, 3], activation='relu', padding='same', strides=(1, 1, 1), name='conv4b')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)
    # 5th block
    x = Conv3D(512, [3, 3, 3], activation='relu', padding='same', strides=(1, 1, 1), name='conv5a')(x)
    x = Conv3D(512, [3, 3, 3], activation='relu', padding='same', strides=(1, 1, 1), name='conv5b')(x)
    x = ZeroPadding3D(padding=(0, 1, 1), name='zeropadding')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')(x)
    # full connection
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(nb_classes, activation='softmax', name='fc8')(x)

    model = Model(input_tensor, output_tensor)
    return model


# f = h5py.File('/home/yitong/venv_yitong/cholec80_phase/params/c3d-sports1M_weights.h5', 'r')
# print(f.keys())
# for key in f.keys():
#     print(f[key])

# model = model_from_json(open('/home/yitong/venv_yitong/cholec80_phase/sports_1M.json', 'r').read())
# model.load_weights('/home/yitong/venv_yitong/cholec80_phase/params/c3d-sports1M_weights.h5')

model = C3Dnet(487, (16, 112, 112, 3))
model.load_weights('/home/yitong/venv_yitong/cholec80_phase/params/conv3d_deepnetA_sport1m_iter_1900000_TF.model')
print('keras')

# py_model = Endo3D()
# model_dict = py_model.state_dict()
# for k,v in model_dict.items():
#     print(k, v.size())
# pretrained_dict = dd.io.load('/home/yitong/venv_yitong/cholec80_phase/params/c3d-sports1M_weights.h5')
# new_pre_dict = {}
# for k, v in pretrained_dict.items():
#     print(k)
#     for k1,v1 in v.items():
#         print(v1.shape)
# new_pre_dict[k] = torch.Tensor(v)
