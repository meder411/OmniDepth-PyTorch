import torch
import torch.nn as nn
import torch.nn.functional as F

class SquaredGradientLoss(nn.Module):
	'''Compute the gradient magnitude of an image using the simple filters as in: 
	Garg, Ravi, et al. "Unsupervised cnn for single view depth estimation: Geometry to the rescue." European Conference on Computer Vision. Springer, Cham, 2016.
	'''

	def __init__(self):

		super(SquaredGradientLoss, self).__init__()

		self.register_buffer('dx_filter', torch.FloatTensor([
				[0,0,0],
				[-0.5,0,0.5],
				[0,0,0]]).view(1,1,3,3))
		self.register_buffer('dy_filter', torch.FloatTensor([
				[0,-0.5,0],
				[0,0,0],
				[0,0.5,0]]).view(1,1,3,3))

	def forward(self, pred, mask):
		dx = F.conv2d(
			pred, 
			self.dx_filter.to(pred.get_device()), 
			padding=1, 
			groups=pred.shape[1])
		dy = F.conv2d(
			pred, 
			self.dy_filter.to(pred.get_device()), 
			padding=1, 
			groups=pred.shape[1])

		error = mask * \
			(dx.abs().sum(1, keepdim=True) + dy.abs().sum(1, keepdim=True))

		return error.sum() / (mask > 0).sum().float()



class L2Loss(nn.Module):

	def __init__(self):

		super(L2Loss, self).__init__()

		self.metric = nn.MSELoss()

	def forward(self, pred, gt, mask):
		error = mask * self.metric(pred, gt)
		return error.sum() / (mask > 0).sum().float()


class MultiScaleL2Loss(nn.Module):

	def __init__(self, alpha_list, beta_list):

		super(MultiScaleL2Loss, self).__init__()

		self.depth_metric = L2Loss()
		self.grad_metric = SquaredGradientLoss()
		self.alpha_list = alpha_list
		self.beta_list = beta_list

	def forward(self, pred_list, gt_list, mask_list):

		# Go through each scale and accumulate errors
		depth_error = 0
		for i in range(len(pred_list)):

			depth_pred = pred_list[i]
			depth_gt = gt_list[i]
			mask = mask_list[i]
			alpha = self.alpha_list[i]
			beta = self.beta_list[i]

			# Compute depth error at this scale
			depth_error += alpha * self.depth_metric(
				depth_pred, 
				depth_gt, 
				mask)
		
			# Compute gradient error at this scale
			depth_error += beta * self.grad_metric(
				depth_pred, 
				mask)
		
		return depth_error
