from PIL import Image
import numpy as np
import os
import pickle

data_dir = "./data/NYU_GT/"
depth_out = "./data/depth_norm/"
image_out = "./data/image_norm/"
trans_out = "./data/gen/trans/"
hazy_out = "./data/gen/hazy/"
hazy_images_out = "./data/gen/hazy_bmp/"
trans_images_out = "./data/gen/trans_bmp/"


# im = Image.open(data_dir+"1_Image_.bmp")
# p = np.array(im)

# # np.set_printoptions(threshold=np.nan)
# # print(p)
# print(p.shape)

def normalize_images(data_dir,depth_out,image_out):
	if not os.path.isdir(data_dir):
		print("Directory to data seems ill defined")
	if not os.path.isdir(depth_out):
		print("Creating depth directory")
		os.makedirs(depth_out)
	if not os.path.isdir(image_out):
		print("Creating image directory")
		os.makedirs(image_out)

	files = sorted(os.listdir(data_dir))
	num_images=int(len(files)/2)
	print("PROCESSING "+str(num_images)+" images and their depths")
	stop_count=4
	for file_path in files:
		# if stop_count<0:
		# 	break
		# stop_count-=1

		im = Image.open(data_dir +file_path)
		#format ex : 1_Image_.bmp or 1_Depth_.bmp
		[ident,filetype,ext]=file_path.split("_")
		arr=np.array(im)
		#normalize
		arr= arr/255.0
		
		print("######PROCESSING "+file_path)
		# print(ident,filetype)
		# print(arr.shape)

		if filetype=="Image":
			if len(arr.shape)!=3:
				print("Mismatch at "+file_path)
			# np.savetxt(image_out+ident+".img",arr)
			np.save(image_out+ident+".img",arr.transpose(2,0,1))
		elif filetype=="Depth":
			if len(arr.shape)!=2:
				print("Mismatch at "+file_path)
			# np.savetxt(depth_out+ident+".dpt",arr)
			np.save(depth_out+ident+".dpt",arr)
		else:
			print("Undefined filetype at "+file_path)
			continue

def keyfunc(string):
	return int(string.split(".")[0])

def generate_haze_transmission(depth_out,image_out,trans_out,hazy_out,hazy_images_out,trans_images_out,beta_count):
	
	if (not os.path.isdir(depth_out)) or (not os.path.isdir(image_out)):
		print("Directory to data seems ill defined")
	if not os.path.isdir(trans_out):
		print("Creating transmission map directory")
		os.makedirs(trans_out)
	if not os.path.isdir(trans_images_out):
		print("Creating transmission map images directory")
		os.makedirs(trans_images_out)
	if not os.path.isdir(hazy_out):
		print("Creating hazy directory directory")
		os.makedirs(hazy_out)
	if not os.path.isdir(hazy_images_out):
		print("Creating hazy images directory")
		os.makedirs(hazy_images_out)


	metadata={}

	img_paths = sorted(os.listdir(image_out),key=keyfunc)
	depth_paths=sorted(os.listdir(depth_out),key=keyfunc)

	
	# print(img_paths[0:20])
	# print(depth_paths[0:20])
	# return
	counter=0
	print("GENERATING HAZY IMAGES ")
	for i in range(0,len(img_paths)):
		img_path=img_paths[i]
		print("#####Processing "+img_path)
		dpt_arr=np.load(depth_out+depth_paths[i])
		# need array to be of the form 3 x __ x ___, but loaded image of the form 
		img_arr=np.load(image_out+img_path)#.transpose(2,0,1)
		ident=int(img_path.split(".")[0])
		# airlight of the form [k,k,k]
		k = np.random.uniform(low=0.7,high=1.0)
		#? uniform airlight ? wasn't the whole point of bilinear ... to have dynamic airlight?
		# how is it going to work if training data doesn't follow those rules
		# alternative : sample non uniform airlight and then smooth it
		airlight = np.zeros(img_arr.shape)+k
		for j in range(0,beta_count):
			beta = np.random.uniform(low=0.5,high=1.5)
			print(beta)
			trans = np.exp((-beta*dpt_arr))
			hazy = trans*img_arr + (1-trans)*airlight
			np.save(trans_out+str(counter),trans)
			np.save(hazy_out+str(counter),hazy)

			hazy_t = hazy*255
			# print(hazy_t.shape)
			# print(hazy_t.transpose(1,2,0).shape)
			im = Image.fromarray(hazy_t.transpose(1,2,0).astype(np.uint8))
			im.save(hazy_images_out+ "hsample"+str(counter)+".bmp")
			im = Image.fromarray((trans*255).astype(np.uint8))
			im.save(trans_images_out+ "tsample"+str(counter)+".bmp")
			metadata[counter]=(beta,k,i)
			counter+=1

		# if counter==10:
		# 	# return
		# 	break

	with open('./data/metadata.pickle', 'wb') as handle:
		pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

def getTrainingData(image_out,trans_out,hazy_out,breakup):
	metadata=None
	with open('./data/metadata.pickle', 'rb') as handle:
		metadata=pickle.load(handle)	
	if not metadata:
		return

	if not os.path.isdir("./data/gen/train"):
		os.makedirs("./data/gen/train")


	j_list= sorted(os.listdir(image_out),key=keyfunc)
	t_list= sorted(os.listdir(trans_out),key=keyfunc)
	i_list= sorted(os.listdir(hazy_out),key=keyfunc)
	shape=np.load(hazy_out+i_list[0]).shape
	num_samples=len(i_list)
	channel=shape[0]
	height=shape[1]
	width=shape[2]
	end_idx=0
	for k in range(0,breakup):
		start_idx=end_idx
		end_idx=int((k+1)*(num_samples/breakup))
		if k==4:
			end_idx=max(end_idx,num_samples)
		size = end_idx-start_idx
		print((size,channel,height,width))
		trainY_j=np.zeros((size,channel,height,width))
		trainY_t=np.zeros((size,height,width))
		trainX=np.zeros((size,channel,height,width))
		
		for i in range(start_idx,end_idx):
			print(i)	
			counter=int(i_list[i].split(".")[0])
			ind = metadata[counter][2]

			I=np.load(hazy_out+i_list[i])
			trainX[i,:,:,:]=I

			T=np.load(trans_out+t_list[i])
			trainY_t[i,:,:]=T

			J=np.load(image_out+j_list[ind])
			trainY_j[i,:,:,:]=J

		np.save('./data/gen/train/trainY_j_'+str(start_idx)+"_"+str(end_idx),trainY_j)
		np.save('./data/gen/train/trainY_t_'+str(start_idx)+"_"+str(end_idx),trainY_t)
		np.save('./data/gen/train/trainX_'+str(start_idx)+"_"+str(end_idx),trainX)







# normalize_images(data_dir,depth_out,image_out)
# generate_haze_transmission(depth_out,image_out,trans_out,hazy_out,hazy_images_out,trans_images_out,1)
getTrainingData(image_out,trans_out,hazy_out,20)



