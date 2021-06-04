import json
import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import visdom
import pandas as pd
import wardmetrics
from wardmetrics.core_methods import eval_events


def phase_f1(seq_true, seq_test):
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
            'f1_avg': sum(f1s) / len(f1s),
            'precision_avg': sum(precisions) / len(precisions),
            'recall_avg': sum(recalls) / len(recalls)}


def ward_evaluation(seq_true, seq_test):
    def sequence_segmentation(seg_sequence):
        phases = np.unique(seg_sequence)
        segments_dict = {}
        for phase in phases:
            single_phase_sequence = np.zeros(seg_sequence.shape)
            index_positive_in_true = np.where(seg_sequence == phase)
            single_phase_sequence[index_positive_in_true] = 1
            starts = []
            ends = []
            last_label = 0
            for i in range(len(single_phase_sequence)):
                current_label = single_phase_sequence[i]
                if current_label == 1 and last_label == 0:
                    starts.append(i)
                if current_label == 0 and last_label == 1:
                    ends.append(i)
                last_label = current_label
            if single_phase_sequence[-1] == 1:
                ends.append(len(single_phase_sequence) - 1)
            segments_list = [(starts[i], ends[i]) for i in range(len(starts)) if starts[i] != ends[i]]
            if segments_list != []:
                segments_dict.update({int(phase): segments_list})
        return segments_dict

    transition_idx = np.where(seq_true == 0)
    seq_true = np.delete(seq_true, transition_idx)
    seq_test = np.delete(seq_test, transition_idx)

    true_dict = sequence_segmentation(seq_true)
    test_dict = sequence_segmentation(seq_test)
    #     event_num_true = 0
    #     for phase in true_dict.keys():
    #         event_num_true += len(true_dict[phase])

    #     event_num_test = 0
    #     for phase in test_dict.keys():
    #         event_num_test += len(test_dict[phase])
    values = np.zeros([11, ])
    for phase in true_dict.keys():
        if phase in test_dict.keys():
            gt_event_scores, det_event_scores, detailed_scores, standard_scores = eval_events(true_dict[phase],
                                                                                              test_dict[phase])
            values += np.array(list(detailed_scores.values()))
        else:
            values[2] += len(true_dict[phase])
    detailed_scores = dict(zip(detailed_scores.keys(), values.astype('int')))
    #     plot_event_analysis_diagram(detailed_scores)
    return detailed_scores


def main(test_model_name):
    # test_model_name = test_model_names[0]
    # test_model_name = 'seq_mode_av.pickle'
    # test_model_name = 'seq_LSTM_90_noised.pickle'
    # test_model_name = 'seq_mode_av.pickle'
    print(test_model_name)
    is_vis = False
    if is_vis:
        vis = visdom.Visdom(env='sequence_tester')
    current_path = os.path.abspath(os.getcwd())
    videos_path = os.path.join(current_path, 'data/sacro_jpg')
    with open(os.path.join(videos_path, 'dataset_div1.json'), 'r') as json_data:
        temp = json.load(json_data)
    test_video_list = temp['test'] + temp['validation'] + temp['train']

    for video_name in test_video_list:
        open_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/results', video_name)

        file = open(os.path.join(open_path, test_model_name), 'rb')
        seq_test = pickle.load(file)
        file.close()

        file = open(os.path.join(open_path, 'seq_true.pickle'), 'rb')
        seq_true = pickle.load(file)
        file.close()

        if video_name == test_video_list[0]:
            detailed_scores = ward_evaluation(np.array(seq_true), np.array(seq_test))
        else:
            detailed_scores.values()
            for key in detailed_scores.keys():
                detailed_scores[key] += ward_evaluation(np.array(seq_true), np.array(seq_test))[key]
    return detailed_scores


test_model_names = ['seq_pred.pickle',
                    'seq_mode_av.pickle',
                    'seq_e3d.pickle',
                    'seq_LSTM_90_m2m.pickle',
                    'seq_LSTM_100_m2m.pickle',
                    'seq_LSTM_100_stan.pickle',
                    'seq_LSTM_100_pred.pickle',
                    'seq_LSTM_100_noised.pickle',
                    'seq_LSTM_90_stan.pickle',
                    'seq_LSTM_90_pred.pickle',
                    'seq_LSTM_90_noised.pickle',
                    'seq_transformer_100_stan.pickle',
                    'seq_transformer_100_pred.pickle',
                    'seq_transformer_100_noised.pickle',
                    'seq_transformer_90_stan.pickle',
                    'seq_transformer_90_pred.pickle',
                    'seq_transformer_90_noised.pickle']
detailed_scores = []
for test_model_name in test_model_names:
    detailed_scores.append(main(test_model_name))
keys = [name.split('.', 1)[0] for name in test_model_names]
df = pd.DataFrame.from_dict(dict(zip(keys, detailed_scores)))
df.to_csv("temp.csv")
