import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import CustomDataset, DataLoader
from model.base_model import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob
from tqdm import tqdm
import argparse
import pdb
import warnings
import time

from collections import deque
import itertools

warnings.filterwarnings("ignore")


def parse_opt():
	parser = argparse.ArgumentParser(description="MPN")
	parser.add_argument("--gpus", nargs="+", type=str, help="gpus")
	parser.add_argument(
		"--batch_size", type=int, default=1, help="batch size for training"
	)
	parser.add_argument(
		"--test_batch_size", type=int, default=1, help="batch size for test"
	)
	parser.add_argument("--h", type=int, default=256, help="height of input images")
	parser.add_argument("--w", type=int, default=256, help="width of input images")
	parser.add_argument("--c", type=int, default=3, help="channel of input images")
	parser.add_argument(
		"--t_length", type=int, default=5, help="length of the frame sequences"
	)
	parser.add_argument(
		"--fdim", type=list, default=[128], help="channel dimension of the features"
	)
	parser.add_argument(
		"--pdim", type=list, default=[128], help="channel dimension of the prototypes"
	)
	parser.add_argument(
		"--psize", type=int, default=10, help="number of the prototypes"
	)
	parser.add_argument(
		"--test_iter", type=int, default=1, help="channel of input images"
	)
	parser.add_argument("--K_hots", type=int, default=0, help="number of the K hots")
	parser.add_argument(
		"--alpha", type=float, default=0.5, help="weight for the anomality score"
	)
	parser.add_argument(
		"--th", type=float, default=0.01, help="threshold for test updating"
	)
	parser.add_argument(
		"--num_workers_test",
		type=int,
		default=8,
		help="number of workers for the test loader",
	)
	parser.add_argument(
		"--dataset_type",
		type=str,
		default="ped2",
		help="type of dataset: ped2, avenue, shanghai",
	)
	parser.add_argument(
		"--dataset_path", type=str, default="data/", help="directory of data"
	)
	parser.add_argument("--model_path", type=str, help="model path")
	parser.add_argument(
		"--video_file", type=str, help="filename of video"
	)
	opt = parser.parse_args()
	return opt


def preprocess_image(image, resize_height, resize_width):
	"""
	Convert image to numpy.ndarray. Notes that the color channels are BGR and the color space
	is normalized from [0, 255] to [-1, 1].

	:param filename: the full path of image
	:param resize_height: resized height
	:param resize_width: resized width
	:return: numpy.ndarray
	"""
	image = cv2.resize(image, (resize_width, resize_height))
	image = image.astype(dtype=np.float32)
	image = (image / 127.5) - 1.0
	# set "channels first" ordering, and add a batch dimension
	image = np.transpose(image, (2, 0, 1))
	image = np.expand_dims(image, 0)
	# return the preprocessed image
	return image


def main(opt):
	torch.manual_seed(2020)

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	if opt.gpus is None:
		gpus = "0"
	else:
		gpus = ",".join([str(i) for i in range(len(opt.gpus))])
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

	torch.backends.cudnn.enabled = (
		True  # make sure to use cudnn for computational performance
	)

	loss_func_mse = nn.MSELoss(reduction="none")

	# Loading the trained model
	model = torch.load(opt.model_path)
	if type(model) is dict:
		model = model["state_dict"]
	model.cuda()
	model.eval()

	anomaly_score_total_list = []
	psnr_list = []
	feature_distance_list = []

	buffer = deque(maxlen=opt.t_length)

	vidcap = cv2.VideoCapture(opt.video_file)
	vidlen = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

	pred_err_buffer = []
	norm_err_buffer = []

	with torch.no_grad():
		pbar = tqdm(
			total=vidlen,
			bar_format="{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]",
		)

		frame_idx = 0
		success, im0 = vidcap.read()
		while success:
			img = im0.copy()
			img = preprocess_image(img, resize_height=opt.h, resize_width=opt.w)
			
			buffer.append(img)

			if len(buffer) == opt.t_length:
				imgs = list(itertools.islice(buffer, 0, opt.t_length))
				# concat images along the channel axis
				imgs = np.concatenate(imgs, axis=1)
				# convert the preprocessed images to a torch tensor and flash them to
				# the GPU
				imgs = torch.from_numpy(imgs)
				imgs = imgs.cuda()

				outputs, fea_loss = model.forward(imgs[:, : 3 * 4], weights=None, train=False)

				mse_imgs = loss_func_mse((outputs[:] + 1) / 2, (imgs[:, -3:] + 1) / 2)

				mse_feas = fea_loss.mean(-1)
				# dismap(mse_imgs, frame_idx, name='pred')
				# visualize_pred_err(im0, frame_idx, mse_imgs.squeeze(dim=0), 256, 256)
				pred_err_buffer.append(mse_imgs.squeeze(dim=0))

				mse_feas = mse_feas.reshape((-1, 1, 256, 256))
				# dismap(mse_feas, frame_idx, name='recon')
				# visualize_recon_err(im0, frame_idx, mse_feas.squeeze(dim=0), 256, 256)
				norm_err_buffer.append(mse_feas.squeeze(dim=0))
				mse_imgs = mse_imgs.view((mse_imgs.shape[0], -1))
				mse_imgs = mse_imgs.mean(-1)
				mse_feas = mse_feas.view((mse_feas.shape[0], -1))
				mse_feas = mse_feas.mean(-1)

				for j in range(len(mse_imgs)):
					psnr_score = psnr(mse_imgs[j].item())
					fea_score = psnr(mse_feas[j].item())
					psnr_list.append(psnr_score)
					feature_distance_list.append(fea_score)
			else:
				pred_err_buffer.append(torch.zeros(img.shape[1:]))
				norm_err_buffer.append(torch.zeros(img.shape[1:]))

			pbar.update(1)
			success, im0 = vidcap.read()
			frame_idx += 1

		pbar.close()

		visualize_pred_err_vid(opt.video_file, pred_err_buffer, 256, 256)
		visualize_recon_err_vid(opt.video_file, norm_err_buffer, 256, 256)

	# Measuring the abnormality score and the AUC
	template = calc(15, 2)
	aa = filter(anomaly_score_list(psnr_list), template, 15)
	bb = filter(anomaly_score_list(feature_distance_list), template, 15)
	anomaly_score_total_list += score_sum(aa, bb, opt.alpha)
	anomaly_score_total = np.asarray(anomaly_score_total_list)
	print(f"Total AUC: {anomaly_score_total}")


if __name__ == "__main__":
	opt = parse_opt()
	main(opt)
