import json
import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import visdom
import pandas as pd
import wardmetrics
from wardmetrics.core_methods import eval_events
import torch


def phase_f1(seq_true, seq_test):
    seq_true = np.array(seq_true)
    seq_pred = np.array(seq_test)
    index = np.where(seq_true == 0)
    seq_true = np.delete(seq_true, index)
    seq_pred = np.delete(seq_pred, index)
    # f1 = f1_score(seq_true,seq_test,labels=[0, 1, 2, 3, 4, 5], average='weighted')
    # f1 = f1_score(seq_true, seq_test)

    correct_pred_eval = (torch.tensor(seq_pred) == torch.tensor(seq_true)).sum().item()
    total_pred_eval = torch.tensor(seq_true).size(0)
    acc = correct_pred_eval / total_pred_eval

    phases, weight = np.unique(seq_true, return_counts=True)
    weight = weight/sum(weight)
    weight = dict(zip(phases, weight))
    f1s = {}
    precisions = {}
    recalls = {}
    f1_micro = 0
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

            f1_micro += f1 * weight[phase]
            f1s.update({phase: f1})
            precisions.update({phase: precision})
            recalls.update({phase: recall})
        else:
            f1s.update({phase: np.NaN})
            precisions.update({phase: np.NaN})
            recalls.update({phase: np.NaN})
    return {'f1': f1s, 'precision': precisions, 'recall': recalls,
            'f1_avg': sum(f1s) / len(f1s),
            'f1_micro': f1_micro, 'acc_micro': acc,
            'precision_avg': sum(precisions) / len(precisions),
            'recall_avg': sum(recalls) / len(recalls)}


def ward_evaluation(seq_true, seq_test):
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

    detail_results_list = []
    cm = np.zeros((7, 7))
    f1_micro = 0
    acc_micro = []
    clip_num = []
    for video_name in test_video_list:
        open_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/results', video_name)

        file = open(os.path.join(open_path, test_model_name), 'rb')
        seq_test = pickle.load(file)
        file.close()

        file = open(os.path.join(open_path, 'seq_true.pickle'), 'rb')
        seq_true = pickle.load(file)
        file.close()

        clip_num.append(len(seq_true))
        if is_vis:
            vis.close(video_name)
            vis.line(X=np.array(range(len(seq_true))),
                     Y=np.column_stack((np.array(seq_test), np.array(seq_true))),
                     win=video_name, update='append',
                     opts=dict(title='Surgical workflow sequential', showlegend=True,
                               legend=['Prediction', 'Ground Truth']))

        results = phase_f1(seq_true, seq_test)
        detail_results_list.append(results)
        f1_micro += results['f1_micro']
        acc_micro.append(results['acc_micro'])
        cm += confusion_matrix(np.array(seq_true), np.array(seq_test), labels=[0, 1, 2, 3, 4, 5, 6])
    print('average clip num:', sum(clip_num)/len(clip_num))
    cm = cm / np.sum(cm, axis=1)[:, None]
    f1_micro = f1_micro / len(test_video_list)
    accuracy_std = np.std(np.array(acc_micro)) * 100
    acc_micro = sum(acc_micro) / len(test_video_list)
    print('f1_micro:', f1_micro)
    print('acc_micro:', acc_micro * 100, accuracy_std)
    vis.heatmap(X=cm, win='heatmap', opts=dict(title='confusion matrix',
                                               rownames=['t0', 't1', 't2', 't3', 't4', 't5', 't_not'],
                                               columnnames=['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p_not']))
    accuracys = np.diag(cm)[1:6]
    # print('accuracys:', accuracys)
    detail_f1 = np.zeros((len(detail_results_list), 5))
    detail_precision = np.zeros((len(detail_results_list), 5))
    detail_recall = np.zeros((len(detail_results_list), 5))
    for idx, item in enumerate(detail_results_list):
        detail_f1[idx, :] = np.array(np.array(list(item['f1'].values())))
        detail_precision[idx, :] = np.array(np.array(list(item['precision'].values())))
        detail_recall[idx, :] = np.array(np.array(list(item['recall'].values())))
    precision_std = np.std(np.nanmean(detail_precision, axis=1)) * 100
    recall_std = np.std(np.nanmean(detail_recall, axis=1)) * 100
    print('f1s:', np.nanmean(detail_f1))
    print('precisions:', np.nanmean(detail_precision) * 100, precision_std)
    print('recalls:', np.nanmean(detail_recall) * 100, recall_std)

    detail_f1_av = np.nanmean(detail_f1, axis=0)
    detail_precision_av = np.nanmean(detail_precision, axis=0)
    detail_recall_av = np.nanmean(detail_recall, axis=0)
    df = pd.DataFrame({'F1-score': detail_f1_av,
                       'Precision': detail_precision_av,
                       'Recall': detail_recall_av,
                       'Accuracy': accuracys},
                      index=[f'Phase{i + 1}' for i in range(5)])
    df.loc['Avg.'] = df.mean()
    df.loc[test_model_name] = df.mean()
    df.to_csv("temp.csv")


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
main(test_model_names[16])
