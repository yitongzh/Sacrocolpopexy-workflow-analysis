import os
import json
import sys
import time
import codecs
import numpy as np
import cv2


def get_annotations():
    annotation_path = '/home/yitong/venv_yitong/sacro_wf_analysis/data/leuven/annotations'
    # get the list of file names, ignore the json files with size lower than 3kb and hidden files
    json_list = [f for f in os.listdir(annotation_path) if not f.startswith('.')
                 and os.path.getsize(os.path.join(annotation_path, f)) / 1025 > 3]
    file_names = [os.path.splitext(json_name)[0] for json_name in json_list]

    # generate a dict of file names and annotations
    data = []
    for json_name in file_names:
        json_path = os.path.join(annotation_path, json_name + '.json')
        with codecs.open(json_path, 'r', encoding='gbk') as json_data:
            temp = json.load(json_data)
            data.append(temp)
    dict_annotation = dict(zip(file_names, data))

    return file_names, dict_annotation


def create_labels(total_frames, dict_timestamps, timestamps_prop, filename):
    label_ls = np.ones([int(total_frames), ]) * 6
    if 'Beginning of (1)' in timestamps_prop:
        s1 = dict_timestamps['Beginning of (1)']
        e1 = dict_timestamps['End of (1)']
        label_ls[s1:e1] = 1

    if 'Beginning of (2)' in timestamps_prop:
        s2 = dict_timestamps['Beginning of (2)']
        e2 = dict_timestamps['End of (2)']
        if 'Pausing (2)' in timestamps_prop:
            p2 = dict_timestamps['Pausing (2)']
            r2 = dict_timestamps['Resuming (2)']
            label_ls[s2:p2] = 2
            label_ls[r2:e2] = 2
        else:
            label_ls[s2:e2] = 2

    if 'Beginning of (3)' in timestamps_prop:
        s3 = dict_timestamps['Beginning of (3)']
        e3 = dict_timestamps['End of (3)']
        if 'Pausing (3)' in timestamps_prop:
            p3 = dict_timestamps['Pausing (3)']
            r3 = dict_timestamps['Resuming (3)']
            label_ls[s3:p3] = 3
            label_ls[r3:e3] = 3
        else:
            label_ls[s3:e3] = 3

    if 'Beginning of (4)' in timestamps_prop:
        s4 = dict_timestamps['Beginning of (4)']
        e4 = dict_timestamps['End of (4)']
        label_ls[s4:e4] = 4

    if 'Beginning of (5)' in timestamps_prop:
        s5 = dict_timestamps['Beginning of (5)']
        e5 = dict_timestamps['End of (5)']
        label_ls[s5:e5] = 5

    phase_beginning = np.min(np.array([s1, s2, s3, s4]))
    if filename == 'e8d3eabc-edd1-414f-9d95-277df742655a':
        phase_end = np.max(np.array([e1, e2, e3, e4]))
    else:
        phase_end = np.max(np.array([e1, e2, e3, e4, e5]))
    label_ls[0:phase_beginning] = 0
    label_ls[phase_end:-1] = label_ls[-1] = 0

    # if 'Pausing (2)' in timestamps_prop:
    #     if 'Pausing (3)' in timestamps_prop:
    #         s1 = dict_timestamps['Beginning of (1)']
    #         e1 = dict_timestamps['End of (1)']
    #
    #         s2 = dict_timestamps['Beginning of (2)']
    #         p2 = dict_timestamps['Pausing (2)']
    #         r2 = dict_timestamps['Resuming (2)']
    #         e2 = dict_timestamps['End of (2)']
    #
    #         s3 = dict_timestamps['Beginning of (3)']
    #         p3 = dict_timestamps['Pausing (3)']
    #         r3 = dict_timestamps['Resuming (3)']
    #         e3 = dict_timestamps['End of (3)']
    #
    #         s4 = dict_timestamps['Beginning of (4)']
    #         e4 = dict_timestamps['End of (4)']
    #
    #         s5 = dict_timestamps['Beginning of (5)']
    #         e5 = dict_timestamps['End of (5)']
    #
    #         label_ls[s1:e1] = 1
    #         label_ls[s2:p2] = 2
    #         label_ls[r2:e2] = 2
    #         label_ls[s3:p3] = 3
    #         label_ls[r3:e3] = 3
    #         label_ls[s4:e4] = 4
    #         label_ls[s5:e5] = 5
    #     else:
    #         s1 = dict_timestamps['Beginning of (1)']
    #         e1 = dict_timestamps['End of (1)']
    #
    #         s2 = dict_timestamps['Beginning of (2)']
    #         p2 = dict_timestamps['Pausing (2)']
    #         r2 = dict_timestamps['Resuming (2)']
    #         e2 = dict_timestamps['End of (2)']
    #
    #         s3 = dict_timestamps['Beginning of (3)']
    #         e3 = dict_timestamps['End of (3)']
    #
    #         s4 = dict_timestamps['Beginning of (4)']
    #         e4 = dict_timestamps['End of (4)']
    #
    #         s5 = dict_timestamps['Beginning of (5)']
    #         e5 = dict_timestamps['End of (5)']
    #
    #         label_ls[s1:e1] = 1
    #         label_ls[s2:p2] = 2
    #         label_ls[r2:e2] = 2
    #         label_ls[s3:e3] = 3
    #         label_ls[s4:e4] = 4
    #         label_ls[s5:e5] = 5
    # else:
    #     if 'Pausing (3)' in timestamps_prop:
    #         s1 = dict_timestamps['Beginning of (1)']
    #         e1 = dict_timestamps['End of (1)']
    #
    #         s2 = dict_timestamps['Beginning of (2)']
    #         e2 = dict_timestamps['End of (2)']
    #
    #         s3 = dict_timestamps['Beginning of (3)']
    #         p3 = dict_timestamps['Pausing (3)']
    #         r3 = dict_timestamps['Resuming (3)']
    #         e3 = dict_timestamps['End of (3)']
    #
    #         s4 = dict_timestamps['Beginning of (4)']
    #         e4 = dict_timestamps['End of (4)']
    #
    #         s5 = dict_timestamps['Beginning of (5)']
    #         e5 = dict_timestamps['End of (5)']
    #
    #         label_ls[s1:e1] = 1
    #         label_ls[s2:e2] = 2
    #         label_ls[s3:p3] = 3
    #         label_ls[r3:e3] = 3
    #         label_ls[s4:e4] = 4
    #         label_ls[s5:e5] = 5
    #     else:
    #         s1 = dict_timestamps['Beginning of (1)']
    #         e1 = dict_timestamps['End of (1)']
    #
    #         s2 = dict_timestamps['Beginning of (2)']
    #         e2 = dict_timestamps['End of (2)']
    #
    #         s3 = dict_timestamps['Beginning of (3)']
    #         e3 = dict_timestamps['End of (3)']
    #
    #         s4 = dict_timestamps['Beginning of (4)']
    #         e4 = dict_timestamps['End of (4)']
    #
    #         s5 = dict_timestamps['Beginning of (5)']
    #         e5 = dict_timestamps['End of (5)']
    #
    #         label_ls[s1:e1] = 1
    #         label_ls[s2:e2] = 2
    #         label_ls[s3:e3] = 3
    #         label_ls[s4:e4] = 4
    #         label_ls[s5:e5] = 5

    p0 = np.shape(np.where(label_ls == 0))[1] / total_frames * 100
    p1 = np.shape(np.where(label_ls == 1))[1] / total_frames * 100
    p2 = np.shape(np.where(label_ls == 2))[1] / total_frames * 100
    p3 = np.shape(np.where(label_ls == 3))[1] / total_frames * 100
    p4 = np.shape(np.where(label_ls == 4))[1] / total_frames * 100
    p5 = np.shape(np.where(label_ls == 5))[1] / total_frames * 100
    p6 = np.shape(np.where(label_ls == 6))[1] / total_frames * 100

    # print('video:', filename,'\n','Total frame number:', int(total_frames))
    # print('Percentage of non-phase: %.2f%%' % p0, np.shape(np.where(label_ls == 0))[1])
    # print('Percentage of phase1: %.2f%%' % p1, np.shape(np.where(label_ls == 1))[1])
    # print('Percentage of phase2: %.2f%%' % p2, np.shape(np.where(label_ls == 2))[1])
    # print('Percentage of phase3: %.2f%%' % p3, np.shape(np.where(label_ls == 3))[1])
    # print('Percentage of phase4: %.2f%%' % p4, np.shape(np.where(label_ls == 4))[1])
    # print('Percentage of phase5: %.2f%%' % p5, np.shape(np.where(label_ls == 5))[1])
    # print('Percentage of transition phase: %.2f%%' % p6, np.shape(np.where(label_ls == 6))[1], '\n')
    return list(label_ls), [p0, p1, p2, p3, p4, p5, p6]


def fgo_10():
    np.random.seed(int(time.time()))
    r = []
    for i in range(10):
        x = np.random.random()
        if x < 0.01:
            r.append(1)
        else:
            r.append(0)
    is_ssr = 0
    if np.sum(r) > 0:
        is_ssr = 1
    return is_ssr


def make_labels_dict(file_names, dict_annotation):
    percentage_profile = []
    label_dict = {}
    av_total_frames = 0
    for file in file_names:
        vid_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/data/leuven/videos', file + '.mp4')
        cap = cv2.VideoCapture(vid_path)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap_duration = total_frames / fps

        timestamps_norm = [int(dict_annotation[file][i]['playback_timestamp'] / 1000 / cap_duration * total_frames)
                           for i in range(len(dict_annotation[file]))]
        timestamps_prop = [(dict_annotation[file][i]['content']).split(')', 1)[0] + ')'
                           for i in range(len(dict_annotation[file]))]
        dict_timestamps = dict(zip(timestamps_prop, timestamps_norm))

        label_ls, percentage = create_labels(total_frames, dict_timestamps, timestamps_prop, file)
        label_dict.update({file: label_ls})
        if file != 'e8d3eabc-edd1-414f-9d95-277df742655a':
            percentage_profile.append(percentage)

    # print('fps of recording: %.2f' % fps)
    average_percentage = (np.sum(np.asarray(percentage_profile), axis=0) / len(percentage_profile))
    # for i in range(7):
    #     if i == 0:
    #         print('Average percentage of non-phase: %.2f%%' % average_percentage[i])
    #     else:
    #         if i == 6:
    #             print('Average percentage of transition phase%i: %.2f%%' % (i, average_percentage[i]))
    #         else:
    #             print('Average percentage of phase%i: %.2f%%' % (i, average_percentage[i]))
    return label_dict


def central_crop(img, obj_size):
    h_l = int(obj_size[0] / 2)
    w_l = int(obj_size[1] / 2)
    h_c = int(np.shape(img)[0] / 2)
    w_c = int(np.shape(img)[1] / 2)
    img = img[h_c - h_l:h_c + h_l, w_c - w_l:w_c + w_l, :]
    return img


class ProgressBar:
    def __init__(self, count=0, total=0, width=50):
        self.count = count
        self.total = total
        self.width = width

    def move(self):
        self.count += 1

    def log(self, s):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        # print(s)
        progress = self.width * self.count / self.total
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('#' * int(progress) + '-' * int((self.width - progress)) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()


def main():
    file_names, dict_annotation = get_annotations()
    phases = ['non_phase', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'transition_phase']

    # make directions for saving frames
    for filename in file_names:
        save_dir = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_jpg', filename)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for phase in phases:
            phase_path = os.path.join(save_dir, phase)
            if not os.path.exists(phase_path):
                os.mkdir(phase_path)

    label_dict = make_labels_dict(file_names, dict_annotation)

    for filename in file_names:
        label_ls = label_dict[filename]
        bar = ProgressBar(total=len(label_ls))
        save_dir = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_jpg', filename)
        vid_path = os.path.join('/home/yitong/venv_yitong/sacro_wf_analysis/data/leuven/videos', filename + '.mp4')
        cap = cv2.VideoCapture(vid_path)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        print(filename)
        print('The total frame number in this video is:', int(total_frames))

        frames_in_phase = [np.shape(np.where(np.array(label_ls) == phase_num))[1] for phase_num in range(len(phases))]
        counters = [0, 0, 0, 0, 0, 0, 0]
        frame_pos_dict = {'total_frames': total_frames}
        for i in range(len(label_ls)):
            _, frame = cap.read()
            phase_path = os.path.join(save_dir, phases[int(label_ls[i])])
            jpg_name = '%s' % str(i + int(label_ls[i]) * 10000000).zfill(8) + '.jpg'

            counters[int(label_ls[i])] += 1
            position_in_phase = counters[int(label_ls[i])] / frames_in_phase[int(label_ls[i])] * 100
            frame_pos_dict.update({jpg_name: position_in_phase})

            frame = central_crop(frame, [1080, 1080])
            cv2.imwrite(os.path.join(phase_path, jpg_name), cv2.resize(frame, (300, 300)))
            bar.log('We have arrived at: ' + str(i + 1))
            bar.move()
        file = open(os.path.join(save_dir, 'frames_in_phase.json'), 'w', encoding='utf-8')
        json.dump(frames_in_phase, file, ensure_ascii=False)
        file = open(os.path.join(save_dir, 'frame_pos_dict.json'), 'w', encoding='utf-8')
        json.dump(frame_pos_dict, file, ensure_ascii=False, indent=1)
        cap.release()


'''
    labels = ['Beginning of (1) Dissection of the promontory',
              'End of (1) Dissection of the promontory',

              'Beginning of (2) Dissection of the right parasigmoidal rectal gutter and vault',
              'Pausing (2) Dissection of the right parasigmoidal gutter and the vaginal vault',
              'Resuming (2) Dissection of the right parasigmoidal gutter and the vaginal vault',
              'End of (2) Dissection of the right parasigmoidal rectal gutter and vault',

              'Beginning of 3) Fixation of the implant to the vault',
              'Pausing (3) Fixation of the implant to the vault',
              'Resuming (3) Fixation of the implant to the vault',
              'End of (3) Fixation of the implant to the vault',

              'Beginning of (4) Fixation of the implant to the promontory',
              'End of (4) Fixation of the implant to the promontory',

              'Beginning of (5) Reperitonealisation',
              'End of (5) Reperitonealisation']
'''

if __name__ == '__main__':
    main()
