import torch
import torch.nn as nn

import numpy as np
import OpenEXR, Imath, array

import os
import os.path as osp
import shutil


def mkdirs(path):
	'''Convenience function to make all intermediate folders in creating a directory'''
	try:
		os.makedirs(path)
	except:
		pass


def xavier_init(m):
	'''Provides Xavier initialization for the network weights and 
	normally distributes batch norm params'''
	classname = m.__class__.__name__
	if (classname.find('Conv2d') != -1) or (classname.find('ConvTranspose2d') != -1):
		nn.init.xavier_normal_(m.weight.data)
		m.bias.data.fill_(0)

def save_checkpoint(state, is_best, filename):
	'''Saves a training checkpoints'''
	torch.save(state, filename)
	if is_best:
		basename = osp.basename(filename) # File basename
		idx = filename.find(basename) # Index where path ends and basename begins
		# Copy the file to a different filename in the same directory
		shutil.copyfile(filename, osp.join(filename[:idx], 'model_best.pth'))


def load_partial_model(model, loaded_state_dict):
	'''Loaded a save model, even if the model is not a perfect match. This will run even if there is are layers from the current network missing in the saved model. 
	However, layers without a perfect match will be ignored.'''
	model_dict = model.state_dict()
	pretrained_dict = {k : v for k,v in loaded_state_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)



def load_optimizer(optimizer, loaded_optimizer_dict, device):
	'''Loads the saved state of the optimizer and puts it back on the GPU if necessary.  Similar to loading the partial model, this will load only the optimization parameters that match the current parameterization.'''
	optimizer_dict = optimizer.state_dict()
	pretrained_dict = {k : v for k,v in loaded_optimizer_dict.items() 
		if k in optimizer_dict and k != 'param_groups'}
	optimizer_dict.update(pretrained_dict)
	optimizer.load_state_dict(optimizer_dict)
	for state in optimizer.state.values():
		for k, v in state.items():
			if torch.is_tensor(v):
				state[k] = v.to(device)



def set_caffe_param_mult(m, base_lr, base_weight_decay):
	'''Function that allows us to assign a LR multiplier of 2 and a decay multiplier of 0 to the bias weights (which is common in Caffe)'''
	param_list = []
	for name, params in m.named_parameters():
		if name.find('bias') != -1:
			param_list.append({'params' : params, 'lr' : 2 * base_lr, 'weight_decay' : 0.0})
		else:
			param_list.append({'params' : params, 'lr' : base_lr, 'weight_decay' : base_weight_decay})
	return param_list
