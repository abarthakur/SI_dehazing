import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceSmoothnessLoss(nn.Module):
	def __init__(self,patch_size=1):
		super(SurfaceSmoothnessLoss,self).__init__()
		self._init_filters()
		self.patch_size=patch_size
	
	def _init_filters(self):
		diff_x=1/8*torch.tensor([[1,0,-1],
							[2,0,-2],
							[1,0,-1]])
		diff_y=diff_x.T
		self.diff_x=diff_x.reshape([1,1,3,3])
		self.diff_y=diff_y.reshape([1,1,3,3])
	
	def forward(self,z):
		assert(len(z.shape)==3)
		z=z.reshape([z.shape[0],1,z.shape[1],z.shape[2]])
		z_x=F.conv2d(z,self.diff_x,padding=1)
		z_y=F.conv2d(z,self.diff_y,padding=1)
		n_z=torch.rsqrt(1+z_x**2+z_y**2)
		n_x=n_z*z_x
		n_y=n_z*z_y
		normal=torch.cat([n_x,n_y,n_z], dim=1)
		loss=0
		for i in range(1,self.patch_size+1):
			for j in range(1,self.patch_size+1):
				dot_prods=normal[:,:,i:,j:] * normal[:,:,:-i,:-j]
				cos_dists=1-torch.sum(dot_prods,axis=1)
				loss+=torch.mean(cos_dists)
		return loss


class SumLoss(nn.Module):
	def __init__(self,scaling_coefficient=100,patch_size=1):
		super(SumLoss, self).__init__()
		self.scaling_coefficient=scaling_coefficient
		self.patch_size=patch_size
		self.mse=nn.MSELoss()
		self.aux_loss=SurfaceSmoothnessLoss(patch_size=patch_size)
		
	def forward(self,z,z_true):
		total_loss=self.mse(z,z_true)
		total_loss+=self.scaling_coefficient * self.aux_loss(z)
		return total_loss


class GradientSimilarityOptimizer:
	
	def __init__(self,model,learning_rate,momentum,l2_weight_decay,scaling_coefficient=100,patch_size=1):
		self.learning_rate=learning_rate
		self.momentum=momentum
		self.weight_decay=l2_weight_decay
		self.model=model
		self.aligned=True
		self.scaling_coefficient=scaling_coefficient
		self.patch_size=patch_size
		self.mse=nn.MSELoss()
		self.aux_loss=SurfaceSmoothnessLoss(patch_size=patch_size)
		self.reset_optimizer()

	def reset_optimizer(self):
		self.optimizer= torch.optim.SGD(self.model.parameters(),lr=self.learning_rate,
									   momentum=self.momentum, weight_decay=self.weight_decay)
		
	def optimize_loss(self,z,z_true):
		# compute losses
		main_loss=self.mse(z,z_true)
		aux_loss=self.scaling_coefficient * self.aux_loss(z)
		# compute gradients
		model=self.model
		optimizer=self.optimizer
		grads=[]
		for i,loss in enumerate([main_loss,aux_loss]):
			grads.append([])
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			for p in model.parameters():
				grads[i].append(p.grad.clone())
		# check alignment of aux loss with main loss
		dot_prod=0
		include=[0]
		for j in range(0,len(grads[0])):
			dot_prod+=torch.sum(grads[i][j]*grads[0][j])
		if dot_prod>0:
			print("aux_loss is in alignment")
			self.aligned=True
			include.append(1)
		else:
			print("aux_loss is NOT in alignment")
			# alignment changed, so reset optimizer state
			if self.aligned:
				self.reset_optimizer()
				optimizer=self.optimizer
			self.aligned=False
		# apply gradients
		optimizer.zero_grad()
		for i in include:
			for j,p in enumerate(model.parameters()):
				p.grad+=grads[i][j]
		optimizer.step()
		return self.aligned,main_loss,aux_loss
