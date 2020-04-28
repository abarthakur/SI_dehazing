import numpy as np

def dehaze_image(hazy_images,tmaps):
	assert(len(hazy_images.shape)==4)
	num_samples=hazy_images.shape[0]
	tmaps_flat=tmaps.reshape(num_samples,-1)
	num_pixels=tmaps_flat.shape[1]
	k=int(0.001*num_pixels)
	darkest_k=np.argpartition(tmaps_flat,kth=k,axis=1)[:,:k]
	del tmaps_flat
	hi_flat=np.mean(hazy_images,axis=3).reshape(num_samples,-1)
	dehazed_list=[]
	for s_idx in range(0,num_samples):
		airlight=np.max(hi_flat[s_idx,darkest_k[s_idx,:]])
		airlight= np.array([airlight,airlight,airlight]).reshape(1,1,3)
		denom=np.maximum(tmaps[s_idx,:,:],0.1)
		numerator=hazy_images[s_idx,:,:,:]- airlight
		dehazed=numerator/np.expand_dims(denom,axis=-1) + airlight
		# clip
		dehazed=np.minimum(np.maximum(dehazed,0),1)
		dehazed_list.append(np.expand_dims(dehazed,axis=0))
	return np.concatenate(dehazed_list,axis=0)