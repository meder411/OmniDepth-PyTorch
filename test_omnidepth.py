import torch
import torch.nn as nn

from omnidepth_trainer import OmniDepthTrainer
from network import *
from dataset import *
from util import mkdirs, set_caffe_param_mult

import os.path as osp

# --------------
# PARAMETERS
# --------------
network_type = 'UResNet' # 'RectNet' or 'UResNet'
experiment_name = 'omnidepth'
input_dir = '/data/meder-ppde/omnidepth/datasets'
val_file_list = '/data/meder-ppde/omnidepth/original_test_split.txt'
checkpoint_dir = osp.join('experiments', experiment_name)
checkpoint_path = None
checkpoint_path = osp.join(checkpoint_dir, 'checkpoint_latest.pth')
num_workers = 4
validation_sample_freq = -1
device_ids = [0,1,2,3]


# -------------------------------------------------------
# Fill in the rest
env = experiment_name
device = torch.device('cuda', device_ids[0])

# UResNet
if network_type == 'UResNet':
	model = UResNet()
	alpha_list = [0.445, 0.275, 0.13]
	beta_list = [0.15, 0., 0.]
# RectNet
elif network_type == 'RectNet':
	model = RectNet()
	alpha_list = [0.535, 0.272]
	beta_list = [0.134, 0.068,]
else:
	assert True, 'Unsupported network type'

# Make the checkpoint directory
mkdirs(checkpoint_dir)


# -------------------------------------------------------
# Set up the training routine
network = nn.DataParallel(
	model.float(),
	device_ids=device_ids).to(device)

val_dataloader = torch.utils.data.DataLoader(
	dataset=OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=val_file_list),
	batch_size=1,
	shuffle=False,
	num_workers=num_workers,
	drop_last=False)

trainer = OmniDepthTrainer(
	experiment_name, 
	network, 
	None, 
	val_dataloader, 
	None, 
	None,
	checkpoint_dir,
	device,
	validation_sample_freq=validation_sample_freq)



trainer.evaluate(checkpoint_path)