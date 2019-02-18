__author__ = "Marc Eder"

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

import torch
import torch.utils.data

import numpy as np
from skimage import io
import OpenEXR, Imath, array

import math
import os.path as osp


class OmniDepthDataset(torch.utils.data.Dataset):
	'''PyTorch dataset module for effiicient loading'''

	def __init__(self, 
		root_path, 
		path_to_img_list):

		# Set up a reader to load the panos
		self.root_path = root_path

		# Create tuples of inputs/GT
		self.image_list = np.loadtxt(path_to_img_list, dtype=str)

		# Max depth for GT
		self.max_depth = 8.0


	def __getitem__(self, idx):
		'''Load the data'''

		# Select the panos to load
		relative_paths = self.image_list[idx]

		# Load the panos
		relative_basename = osp.splitext((relative_paths[0]))[0]
		basename = osp.splitext(osp.basename(relative_paths[0]))[0]
		rgb = self.readRGBPano(osp.join(self.root_path, relative_paths[0]))
		depth = self.readDepthPano(osp.join(self.root_path, relative_paths[1]))
		depth_mask = ((depth <= self.max_depth) & (depth > 0.)).astype(np.uint8)

		# Threshold depths
		depth *= depth_mask

		# Make a list of loaded data
		pano_data = [rgb, depth, depth_mask, basename]

		# Convert to torch format
		pano_data[0] = torch.from_numpy(pano_data[0].transpose(2,0,1)).float()
		pano_data[1] = torch.from_numpy(pano_data[1][None,...]).float()
		pano_data[2] = torch.from_numpy(pano_data[2][None,...]).float()

		# Return the set of pano data
		return pano_data
		
	def __len__(self):
		'''Return the size of this dataset'''
		return len(self.image_list)

	def readRGBPano(self, path):
		'''Read RGB and normalize to [0,1].'''
		rgb = io.imread(path).astype(np.float32) / 255.
		return rgb


	def readDepthPano(self, path):
		return self.read_exr(path)[...,0].astype(np.float32)


	def read_exr(self, image_fpath):
		f = OpenEXR.InputFile( image_fpath )
		dw = f.header()['dataWindow']
		w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)    
		im = np.empty( (h, w, 3) )

		# Read in the EXR
		FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
		channels = f.channels( ["R", "G", "B"], FLOAT )
		for i, channel in enumerate( channels ):
			im[:,:,i] = np.reshape( array.array( 'f', channel ), (h, w) )
		return im
