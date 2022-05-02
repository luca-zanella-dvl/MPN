import operator
from pathlib import Path
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def filter(data, template, radius=5):
    arr=np.array(data)
    length=arr.shape[0]  
    newData=np.zeros(length) 

    for j in range(radius//2,arr.shape[0]-radius//2):
        t=arr[ j-radius//2:j+radius//2+1]
        a=np.multiply(t,template)
        newData[j]=a.sum()
    # expand
    for i in range(radius//2):
        newData[i]=newData[radius//2]
    for i in range(-radius//2,0):
        newData[i]=newData[-radius//2]    
    # import pdb;pdb.set_trace()
    return newData

def calc(r=5, sigma=2):
    k = np.zeros(r)
    for i in range(r):
        k[i] = 1/((2*math.pi)**0.5*sigma)*math.exp(-((i-r//2)**2/2/(sigma**2)))
    return k

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):

    return 10 * math.log10(1 / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def dismap(x, frame_idx, name='dismap'):
    # import pdb;pdb.set_trace()
    x = x.data.cpu().numpy()
    x = x.mean(1)
    for j in range(x.shape[0]):
        plt.cla()
        y = x[j]
        # import pdb;pdb.set_trace()
        df = pd.DataFrame(y)
        sns.heatmap(df)
        plt.savefig('results/dismap/{}_{}.png'.format(name,str(frame_idx + j)))
        plt.close()
    return True

def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re

def point_score(outputs, imgs):
    
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score
    
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc

def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

def draw_score_curve(aa, bb, cc, cur_gt, name='results/curves_pt', vid = ''):
    
    T = range(len(aa))
    xnew = np.linspace(0,len(aa),10*len(aa))
    aa_new = 1-np.array(aa)
    aa_new = moving_average(aa_new,5)
    bb_new = 1-np.array(bb)
    bb_new = moving_average(bb_new,5)
    cc_new = 1-np.array(cc)
    cc_new = moving_average(cc_new,5)
    # cur_gt = make_interp_spline(T, cur_gt)(xnew)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cur_ans = cc_new
    # print(vid)
    # import pdb;pdb.set_trace()
    ax1.plot(cur_gt, color='r')
    ax2.plot(cur_ans, color='g')
    plt.title(vid)
    plt.show()
    plt.savefig(name+'/'+vid+'_all.png')
    # print('Save: ',root +'/'+vid+'.png')
    plt.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cur_ans = aa_new
    # print(vid)
    # import pdb;pdb.set_trace()
    ax1.plot(cur_gt, color='r')
    ax2.plot(cur_ans, color='g')
    plt.title(vid)
    plt.show()
    plt.savefig(name+'/'+vid+'_fra.png')
    # print('Save: ',root +'/'+vid+'.png')
    plt.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cur_ans = bb_new
    # print(vid)
    # import pdb;pdb.set_trace()
    ax1.plot(cur_gt, color='r')
    ax2.plot(cur_ans, color='g')
    plt.title(vid)
    plt.show()
    plt.savefig(name+'/'+vid+'_fea.png')
    # print('Save: ',root +'/'+vid+'.png')
    plt.close()
    # import pdb;pdb.set_trace()
    return True

def depict(videos_list, psnr_list, feature_distance_list, labels_list, root='results/AUCs'):
    video_num = 0
    label_length = 0
    import pdb
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        start = label_length
        end = label_length + len(psnr_list[video_name])
        # anomaly_score_total_list = score_sum(anomaly_score_list(psnr_list[video_name]), 
        #                                  anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)
        anomaly_score_ae_list = np.asarray(anomaly_score_list(psnr_list[video_name]))
        anomaly_score_mem_list = np.asarray(anomaly_score_list_inv(feature_distance_list[video_name]))
        if (1-labels_list[start:end]).max() <1 or (1-labels_list[start:end]).min()==1:
            accuracy_ae = accuracy_me = 0
        else:
            accuracy_ae = AUC(anomaly_score_ae_list, np.expand_dims(1-labels_list[start:end], 0))
            accuracy_me = AUC(anomaly_score_mem_list, np.expand_dims(1-labels_list[start:end], 0))
        assert len(labels_list[start:end])==len(anomaly_score_ae_list)
        # pdb.set_trace()
        label_length = end
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        #print vid, tf_idf
        ax1.plot(1-labels_list[start:end], color='r')
        ax2.plot(anomaly_score_ae_list, color='g')
        ax3.plot(anomaly_score_mem_list, color='b')
        plt.title(video_name+' {:.4f} {:.4f}'.format(accuracy_ae, accuracy_me), y=3.4)
        plt.show()
        if not os.path.exists(root):
            os.makedirs(root)
        plt.savefig(root+'/'+video_name+'.png')
        # print('Save: ',root +'/'+vid+'.png')
        plt.close()
    # pdb.set_trace()
    return True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize_frame_with_text(f_name, score, output_dir="output"):
    filename, file_extension = os.path.splitext(f_name)
    #f_idx = int(filename.split("/")[-1]) 
    f_idx= int(f_name.split("/")[-1].split(".")[-2])
    frame = cv2.imread(f_name)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"{f_idx:06}{file_extension}")

    if score >= 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        threshold = 0.5
        bgr_red = (0, 0, 255)
        bgr_green = (0, 255, 0)
        # font_color = bgr_red if score > threshold else bgr_green
        line_type = 5

        ano_description = "Anomaly" if score < threshold else "Normal"
        font_color = bgr_red if ano_description == "Anomaly" else bgr_green
        
        height, _, _ = frame.shape
        margin = 10
        bottom_left_corner_of_text = (margin, height - margin)

        cv2.putText(
            frame,
            f"{score:.2f}",
            bottom_left_corner_of_text,
            font,
            font_scale,
            font_color,
            line_type,
        )

        text_width, text_height = cv2.getTextSize("{:.2f}".format(score), font, font_scale, line_type)[0]
        ano_corner = (margin * 2 + text_width, height - margin)

        cv2.putText(
            frame,
            ano_description,
            ano_corner,
            font,
            font_scale,
            font_color,
            line_type,
        )

    # Display the image
    # cv2.imshow("frame", frame)
    cv2.imwrite(output_path, frame)
    cv2.waitKey(0)


def visualize_frame_with_text_vid(vid_file, anomaly_score_total, t_length, output_dir="results/vid"):
    vidcap = cv2.VideoCapture(vid_file)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, 'anomaly_vid.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    f_idx = 0
    success, frame = vidcap.read()
    while success:
        if f_idx >= t_length - 1:
            anomaly_score = anomaly_score_total[f_idx - (t_length - 1)]
        else:
            anomaly_score = -1
            
        if anomaly_score >= 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            anomaly_thres = 0.5
            # font_color = bgr_red if score > threshold else bgr_green
            thickness=2
            
            text_scale = max(font_scale, frame.shape[1] / 1600.0)
            top_left_corner_of_text = (0, int(30 * text_scale))

            anomaly_text = "Anomalous" if anomaly_score < anomaly_thres else "Normal"
            color = (0, 0, 255) if anomaly_text == "Anomalous" else (0, 255, 0)

            cv2.putText(
                frame,
                f"frame: {f_idx} event: {anomaly_text} score: {anomaly_score:.2f}",
                top_left_corner_of_text,
                font,
                text_scale,
                color,
                thickness=thickness,
            )

        out.write(frame)
        success, frame = vidcap.read()
        f_idx += 1
    
    cv2.destroyAllWindows()
    vidcap.release()


# def visualize_pred_err(f_name, pred_err, resize_width, resize_height, output_dir="output"):
#     filename, file_extension = os.path.splitext(f_name)
#     #f_idx = int(filename.split("/")[-1])
#     f_idx = int(f_name.split("/")[-1].split(".")[-2])
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     output_path = os.path.join(output_dir, f"{f_idx:06}{file_extension}")
#     img = cv2.imread(f_name)
#     img_resized = cv2.resize(img, (resize_width, resize_height))
#     heatmap_img = cv2.applyColorMap(pred_err, cv2.COLORMAP_JET)
#     fin = cv2.addWeighted(heatmap_img, 0.7, img_resized, 0.3, 0)
#     # Display the image
#     #cv2.imshow("frame", fin)
#     cv2.imwrite(output_path, fin)
#     cv2.waitKey(0)


def visualize_pred_err(im0, f_idx, pred_err, resize_width, resize_height, output_dir="results/pred"):
    pred_err = pred_err.data.cpu().numpy()
    pred_err = pred_err.mean(0)
    pred_err *= 255.0/pred_err.max()
    pred_err = pred_err.astype(np.uint8)
    pred_err = np.expand_dims(pred_err, axis=0)
    pred_err = np.transpose(pred_err, (1, 2, 0))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"{f_idx:06}.png")
    img_resized = cv2.resize(im0, (resize_width, resize_height))
    heatmap_img = cv2.applyColorMap(pred_err, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, 0.7, img_resized, 0.3, 0)
    # Display the image
    #cv2.imshow("frame", fin)
    cv2.imwrite(output_path, fin)
    # cv2.waitKey(0)

def visualize_recon_err(im0, f_idx, recon_err, resize_width, resize_height, output_dir="results/recon"):
    recon_err = recon_err.data.cpu().numpy()
    recon_err = recon_err.mean(0)
    recon_err *= 255.0/recon_err.max()
    recon_err = recon_err.astype(np.uint8)
    recon_err = np.expand_dims(recon_err, axis=0)
    recon_err = np.transpose(recon_err, (1, 2, 0))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"{f_idx:06}.png")
    img_resized = cv2.resize(im0, (resize_width, resize_height))
    heatmap_img = cv2.applyColorMap(recon_err, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, 0.7, img_resized, 0.3, 0)
    # Display the image
    #cv2.imshow("frame", fin)
    cv2.imwrite(output_path, fin)
    # cv2.waitKey(0)

def visualize_pred_err_vid(vid_file, pred_err_buffer, resize_width, resize_height, output_dir="results/vid/pred"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, 'pred_err_vid.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 12.5, (resize_width,resize_height))

    max_val = max([pred_err.data.cpu().numpy().mean(0).max() for pred_err in pred_err_buffer])

    vidcap = cv2.VideoCapture(vid_file)
    f_idx = 0
    success, im0 = vidcap.read()
    while success:
        pred_err = pred_err_buffer[f_idx].data.cpu().numpy()
        pred_err = pred_err.mean(0)
        pred_err *= 255.0/max_val
        pred_err = pred_err.astype(np.uint8)
        pred_err = np.expand_dims(pred_err, axis=0)
        pred_err = np.transpose(pred_err, (1, 2, 0))
        img_resized = cv2.resize(im0, (resize_width, resize_height))
        heatmap_img = cv2.applyColorMap(pred_err, cv2.COLORMAP_JET)
        fin = cv2.addWeighted(heatmap_img, 0.7, img_resized, 0.3, 0)
        # Display the image
        #cv2.imshow("frame", fin)
        out.write(fin)

        success, im0 = vidcap.read()
        f_idx += 1

def visualize_recon_err_vid(vid_file, recon_err_buffer, resize_width, resize_height, output_dir="results/vid/recon"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, 'recon_err_vid.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 12.5, (resize_width,resize_height))

    max_val = max([pred_err.data.cpu().numpy().mean(0).max() for pred_err in recon_err_buffer])

    vidcap = cv2.VideoCapture(vid_file)
    f_idx = 0
    success, im0 = vidcap.read()
    while success:
        recon_err = recon_err_buffer[f_idx].data.cpu().numpy()
        recon_err = recon_err.mean(0)
        recon_err *= 255.0/max_val
        recon_err = recon_err.astype(np.uint8)
        recon_err = np.expand_dims(recon_err, axis=0)
        recon_err = np.transpose(recon_err, (1, 2, 0))
        img_resized = cv2.resize(im0, (resize_width, resize_height))
        heatmap_img = cv2.applyColorMap(recon_err, cv2.COLORMAP_JET)
        fin = cv2.addWeighted(heatmap_img, 0.7, img_resized, 0.3, 0)
        # Display the image
        #cv2.imshow("frame", fin)
        out.write(fin)

        success, im0 = vidcap.read()
        f_idx += 1
