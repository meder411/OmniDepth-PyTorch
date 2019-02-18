import torch
import torch.nn as nn

import visdom

from omnidepth_trainer import OmniDepthTrainer
from network import *
from criteria import *
from dataset import *
from util import mkdirs, set_caffe_param_mult

import os.path as osp

# --------------
# PARAMETERS
# --------------
network_type = 'UResNet' # 'RectNet' or 'UResNet'
experiment_name = 'omnidepth'
input_dir = '' # Dataset location
train_file_list = '' # File with list of training files
val_file_list = '' # File with list of validation files
checkpoint_dir = osp.join('experiments', experiment_name)
checkpoint_path = None
# checkpoint_path = osp.join(checkpoint_dir, 'checkpoint_latest.pth')
load_weights_only = False
batch_size = 10
num_workers = 4
lr = 2e-4
step_size = 3
lr_decay = 0.5
num_epochs = 10
validation_freq = 1
visualization_freq = 5
validation_sample_freq = -1
device_ids = [0,1,2,3]


# -------------------------------------------------------
# Fill in the rest
vis = visdom.Visdom()
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

train_dataloader = torch.utils.data.DataLoader(
	dataset=OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=train_file_list),
	batch_size=batch_size,
	shuffle=True,
	num_workers=num_workers,
	drop_last=True)

val_dataloader = torch.utils.data.DataLoader(
	dataset=OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=val_file_list),
	batch_size=batch_size,
	shuffle=False,
	num_workers=num_workers,
	drop_last=True)

criterion = MultiScaleL2Loss(alpha_list, beta_list)

# Set up network parameters with Caffe-like LR multipliers
param_list = set_caffe_param_mult(network, lr, 0)
optimizer = torch.optim.Adam(
	params=param_list,
	lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
	step_size=step_size,
	gamma=lr_decay)

trainer = OmniDepthTrainer(
	experiment_name, 
	network, 
	train_dataloader, 
	val_dataloader, 
	criterion, 
	optimizer,
	checkpoint_dir,
	device,
	visdom=[vis, env],
	scheduler=scheduler, 
	num_epochs=num_epochs,
	validation_freq=validation_freq,
	visualization_freq=visualization_freq, 
	validation_sample_freq=validation_sample_freq)



trainer.train(checkpoint_path, load_weights_only)