import torch
import torch.nn as nn
import math


def get_basic_block(in_channels,out_channels,conv_kernel_size):
	# pytorch has no easy way to do asymmetric padding 
	# which is needed for a SAME padding constraint for even kernels
	assert(conv_kernel_size%2==1)
	padding=math.floor(conv_kernel_size/2)
	conv_layer=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
						kernel_size=conv_kernel_size,stride=1,
						padding=padding,bias=True)
	nn.init.xavier_normal_(conv_layer.weight)
	conv_layer.bias.data.fill_(0.01)
	return nn.Sequential(
		conv_layer,
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
		nn.Upsample(scale_factor=2,mode="nearest"),
	)

class CoarseNet(nn.Module):
	
	def __init__(self):
		super(CoarseNet, self).__init__()
		self.layers=nn.Sequential(
						get_basic_block(3,5,11),
						get_basic_block(5,5,9),
						get_basic_block(5,10,7)
						)
		self.final=nn.Sequential(nn.Linear(10,1,True),
								nn.Sigmoid()
								)
 
	def forward(self,x):
		# in original format B x H X W X C
		assert(x.shape[3]==3)
		# permute to B x C x W x H
		x = x.permute(0,3,2,1)
		tx=self.layers(x)
		tx=tx.permute(0,2,3,1)
		tx=self.final(tx)
		tx=tx.squeeze(3)
		# permute from B x W x H to B x H x W
		tx = tx.permute(0,2,1)
		return tx
