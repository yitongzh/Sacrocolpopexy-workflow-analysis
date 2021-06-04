import json
import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import visdom


def phase_f1(seq_true, seq_test):
    accuracy = sum([seq_test[i] == seq_true[i] for i in range(len(seq_test))]) / len(seq_test) * 100
    seq_true = np.array(seq_true)
    seq_pred = np.array(seq_test)
    index = np.where(seq_true == 0)
    seq_true = np.delete(seq_true, index)
    seq_pred = np.delete(seq_pred, index)
    # f1 = f1_score(seq_true,seq_test,labels=[0, 1, 2, 3, 4, 5], average='weighted')
    # f1 = f1_score(seq_true, seq_test)

    phases = np.unique(seq_true)
    f1s = {}
    precisions = {}
    recalls = {}
    for phase in range(1, 6):
        if phase in phases:
            index_positive_in_true = np.where(seq_true == phase)
            index_positive_in_pred = np.where(seq_pred == phase)
            index_negative_in_true = np.where(seq_true != phase)
            index_negative_in_pred = np.where(seq_pred != phase)

            a = seq_true[index_positive_in_pred]
            unique, counts = np.unique(a, return_counts=True)
            count_dict = dict(zip(unique, counts))
            if phase in count_dict.keys():
                tp = count_dict[phase]
            else:
                tp = 0
            fp = len(index_positive_in_pred[0]) - tp

            b = seq_true[index_negative_in_pred]
            unique, counts = np.unique(b, return_counts=True)
            count_dict = dict(zip(unique, counts))
            if phase in count_dict.keys():
                fn = count_dict[phase]
            else:
                fn = 0
            tn = len(index_negative_in_pred[0]) - fn

            f1 = tp / (tp + 0.5 * (fp + fn))
            if tp != 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
            else:
                precision = 0
                recall = 0

            f1s.update({phase: f1})
            precisions.update({phase: precision})
            recalls.update({phase: recall})
        else:
            f1s.update({phase: np.NaN})
            precisions.update({phase: np.NaN})
            recalls.update({phase: np.NaN})

    return {'f1': f1s, 'precision': precisions, 'recall': recalls,
            'f1_avg': sum(f1s.values()) / len(f1s),
            'precision_avg': sum(precisions) / len(precisions),
            'recall_avg': sum(recalls) / len(recalls),
            'accuracy': accuracy}


vis = visdom.Visdom(env='sequence_tester')
current_path = os.path.abspath(os.getcwd())
videos_path = os.path.join(current_path, 'data/sacro_jpg')
with open(os.path.join(videos_path, 'dataset_div1.json'), 'r') as json_data:
    temp = json.load(json_data)
test_video_list = temp['test'] + temp['validation'] + temp['train']

detail_results_list = []
cm = np.zeros((7, 7))
lengths = []
for video_name in test_video_list:
    open_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/results', video_name)

    file = open(os.path.join(open_path, 'seq_e3d.pickle'), 'rb')
    seq_e3d = pickle.load(file)
    file.close()
    lengths.append(len(seq_e3d))

    file = open(os.path.join(open_path, 'seq_true.pickle'), 'rb')
    seq_true = pickle.load(file)
    file.close()

    file = open(os.path.join(open_path, 'seq_LSTM_100_m2m.pickle'), 'rb')
    seq_slin = pickle.load(file)
    file.close()

    detail_results_list.append({'e3d': phase_f1(seq_true, seq_e3d), 'slin': phase_f1(seq_true, seq_slin)})

lengths = np.array(lengths)
sorted_idx = np.argsort(lengths)
# lengths = np.sort(lengths)

vis.close('accuracy')
for idx in sorted_idx:
    video_len = lengths[idx]
    e3d_accuracy = detail_results_list[idx]['e3d']['accuracy']
    slin_accuracy = detail_results_list[idx]['slin']['accuracy']
    vis.line(X=np.array([video_len]), Y=np.column_stack((np.array(e3d_accuracy), np.array(slin_accuracy))),
             win='accuracy', update='append', opts=dict(title='accuracy', showlegend=True, legend=['e3d', 'slinm2m']))

