#! /usr/bin/env python
import sys
sys.path.append('../')
from sacro_wf_analysis.model import Endo3D
import torch
import torchvision
import torchvision.transforms as transforms
#from utils.UCF101 import UCF101
import pickle
import scipy.io as scio
from sacro_loder import sacro_loder


def main():

    # list_file = open('list1.pickle', 'wb')
    # pickle.dump(fi_A, list_file)
    # list_file.close()

    # list_file = open('./data/UCF101-16-112-112/train_set1.pickle', 'rb')
    # train_set1 = pickle.load(list_file)
    # list_file.close()
    #
    # list_file = open('./data/UCF101-16-112-112/train_set2.pickle', 'rb')
    # train_set2 = pickle.load(list_file)
    # list_file.close()
    #
    # list_file = open('./data/UCF101-16-112-112/train_labels1.pickle', 'rb')
    # train_labels1 = pickle.load(list_file)
    # list_file.close()
    #
    # list_file = open('./data/UCF101-16-112-112/train_labels2.pickle', 'rb')
    # train_labels2 = pickle.load(list_file)
    # list_file.close()
    #
    # print((train_set1[103]).size())
    # print(len(train_set1))
    # print((train_set2[103]).size())
    # print(len(train_set2))
    sacro = sacro_loder(building_block=False)
    _, clip = sacro.get_clip()


    data_path = 'example_output.mat'
    # scio.savemat(data_path, {'train_set': (sacro[3][1]).numpy(),
    #                          'train_labels': (sacro[3][0]).numpy(),
    #                          'validation_set': sacro.validation_inputs.numpy(),
    #                          'validation_labels': sacro.validation_labels.numpy()})

    scio.savemat(data_path, {'train_set': clip*255})




if __name__ == '__main__':
    main()
