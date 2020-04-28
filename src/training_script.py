import torch 
import numpy as np
import time
from smoothness import SumLoss,GradientSimilarityOptimizer


def training_loop(model,dataset,num_epochs,checkpoint,run_dir,
					grad_sim,scaling_coefficient,patch_size,
					batch_size,lr_initial,lr_decay_factor,lr_decay_interval,
					momentum,l2_weight_decay,training_callback=None):
	# create log file
	logfilename=run_dir+"/log.txt"
	num_samples=dataset["hazy_image"].shape[0]
	total_time=0
	epoch_losses=[]
	# epoch loop
	for epoch in range(0,num_epochs):
		print("Epoch # "+str(epoch))
		if epoch%lr_decay_interval==0:
			if epoch==0:
				learning_rate=lr_initial
			else:
				learning_rate=learning_rate*lr_decay_factor
			if grad_sim:
				optimizer=GradientSimilarityOptimizer(model,learning_rate,momentum,
												l2_weight_decay,scaling_coefficient,patch_size)
			else:
				optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,
											momentum=momentum, weight_decay=l2_weight_decay)
		# shuffle examples
		shuffle=np.random.permutation(num_samples)
		start_batch=0
		loss_values=[]
		minibatch_num=0
		epoch_start_time=time.time()
		# minibatch loop
		while start_batch<num_samples:
			end_batch=min(start_batch+batch_size,num_samples)
			mb_idcs=shuffle[start_batch:end_batch]
			# sorted order required for h5py
			mb_idcs=np.sort(mb_idcs)
			h_batch=torch.from_numpy(dataset["hazy_image"][mb_idcs,:,:,:])
			t_batch=torch.from_numpy(dataset["trans_map"][mb_idcs,:,:])
			# step
			pred_batch=model(h_batch)
			if grad_sim:
				aligned,main_loss,aux_loss=optimizer.optimize_loss(pred_batch,t_batch)
				with open(logfilename,"a+") as fi:
					fi.write("epoch="+str(epoch)+",")
					fi.write("minibatch="+str(minibatch_num)+",")
					fi.write("aligned="+str(aligned)+",")
					fi.write("aux_loss="+str(aux_loss.detach().numpy())+"\n")
				loss_batch=main_loss.detach().numpy()
			else:
				sum_loss=SumLoss(scaling_coefficient,patch_size)
				loss_batch=sum_loss(pred_batch,t_batch)
				optimizer.zero_grad()
				loss_batch.backward(retain_graph=True)
				optimizer.step()
				loss_batch=loss_batch.detach().numpy()
			
			print("Epoch # "+str(epoch)+", Minibatch # "+str(minibatch_num)+
				", Loss : ",loss_batch)

			with open(logfilename,"a+") as fi:
					fi.write("epoch="+str(epoch)+",")
					fi.write("minibatch="+str(minibatch_num)+",")
					fi.write("main_loss="+str(loss_batch)+"\n")
			start_batch+=batch_size
			minibatch_num+=1
			loss_values.append(loss_batch)
		
		epoch_time=time.time()-epoch_start_time
		total_time+=epoch_time
		etc=(total_time/(epoch+1)) * (num_epochs-epoch-1)
		etc/=60*60
		loss_mean=np.mean(np.array(loss_values))
		epoch_losses.append(loss_mean)
		print("Epoch # "+str(epoch+1)+" Loss :",loss_mean," Time (mins) :",epoch_time/60," ETC (hrs):",etc)
		if (epoch+1)%checkpoint==0:
			torch.save(model,run_dir+"/model_"+str(epoch+1))

		if training_callback:
			training_callback(model,dataset)



if __name__=="__main__":
	import h5py
	import os
	import sys
	from mymodels import CoarseNet

	args={
		"num_epochs":50,
		"checkpoint":10,
		"grad_sim":False,
		"scaling_coefficient":100,
		"patch_size":1,
		"batch_size":100,
		"lr_initial":0.01,
		# for our experiments we use a fixed LR
		"lr_decay_factor":0.1,
		"lr_decay_interval":100,
		"momentum":0.9,
		"l2_weight_decay":5e-04
		}

	for i in range(1,len(sys.argv)):
		string=sys.argv[i]
		parts=string.split("=")
		assert(len(parts)==2)
		if parts[0]=="run_dir":
			dir_path=parts[1]
			if os.path.exists(dir_path):
				assert(os.path.isdir(dir_path))
			else:
				os.makedirs(dir_path)
			args["run_dir"]=parts[1]
		elif parts[0] in ["num_epochs","checkpoint"]:
			args[parts[0]]=int(parts[1])
		elif parts[0]=="patch":
			args["patch_size"]=int(parts[1])
		elif parts[0]=="scaling":
			args["scaling_coefficient"]=float(parts[1])
		elif parts[0]=="grad_sim":
			args["grad_sim"]=bool(int(parts[1]))

	logfilename=args["run_dir"]+"/log.txt"
	with open(logfilename,"a+") as fi:
		fi.write(str(args)+"\n")

	dataset = h5py.File("../data/nyu_hazy_trn.mat","r")
	model=CoarseNet()
	training_loop(model=model,dataset=dataset,**args)
