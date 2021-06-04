import cv2
import numpy as np
import torch
from model import Endo3D
from tqdm import tqdm
import os
import pickle


def central_crop(img, obj_size):
    h_l = int(obj_size[0] / 2)
    w_l = int(obj_size[1] / 2)
    h_c = int(np.shape(img)[0] / 2)
    w_c = int(np.shape(img)[1] / 2)
    img = img[h_c - h_l:h_c + h_l, w_c - w_l:w_c + w_l, :]
    return img


# video_path = '/home/yitong/venv_yitong/sacro_wf_analysis/data/leuven/videos/becb53be-978f-4233-bbb1-ed854f48dc21'
video_path = '/home/yitong/venv_yitong/sacro_wf_analysis/data/leuven/videos/d6994bf0-b53c-49c6-8043-c485fd847e4a'
# save_path = '/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_jpg/becb53be-978f-4233-bbb1-ed854f48dc21'
save_path = '/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_jpg/d6994bf0-b53c-49c6-8043-c485fd847e4a'
cap = cv2.VideoCapture(video_path)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
counter = 0
clip = np.zeros([16, 112, 112, 3])
seq_pre = []
fc_list = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Endo3D().to(device)
model.load_state_dict(torch.load('./params/cross_validation/div3/params_endo3d.pkl'))

name_idx = 0
frame_ls = [i for i in range(int(total_frames)) if i % 10 == 0]
for i in tqdm(frame_ls, ncols=80):
    # if counter == 16:
    #     input_clip = torch.from_numpy(np.float32(clip.transpose((3, 0, 1, 2)))).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         output, x_fc = model.forward_cov(input_clip)
    #     _, predicted_labels = torch.max(output.cpu().data, 1)
    #     seq_pre.append(predicted_labels.numpy()[0])
    #     fc_list.append(x_fc[0, :].cpu())
    #     del output, x_fc
    #     torch.cuda.empty_cache()
    #     counter = 0
    #     clip = np.zeros([16, 112, 112, 3])
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    _, frame = cap.read()
    frame = central_crop(frame, [frame.shape[0], frame.shape[0]])
    # frame = frame[:, :, -1::-1]
    # frame = cv2.resize(frame, (112, 112))

    #######
    jpg_name = '%s' % str(name_idx).zfill(8) + '.jpg'
    cv2.imwrite(os.path.join(save_path, jpg_name), cv2.resize(frame, (300, 300)))
    name_idx += 1
    #######
    # result = np.zeros(np.shape(frame), dtype=np.float32)
    # frame = cv2.normalize(frame, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # clip[counter, :, :, :] = frame
    counter += 1



# save_path = '/home/yitong/venv_yitong/sacro_wf_analysis/data/sacro_sequence/whole/test/d6994bf0-b53c-49c6-8043-c485fd847e4a'
# a = open(os.path.join(save_path, 'seq_pred.pickle'), 'wb')
# pickle.dump(seq_pre, a)
# a.close()
#
# c = open(os.path.join(save_path, 'fc_list.pickle'), 'wb')
# pickle.dump(fc_list, c)
# c.close()