## Project Overview

This project proposes a novel auxiliary loss for the task of transmission estimation in CNN based single image dehazing methods. This auxiliary loss models surface smoothness by aligning the normals within a small patch with the objective of removing surface artifacts from appearing on the transmission map, such as the one shown below as it hinders haze removal.


## Project Description

* Haze is a common atmospheric phenomenon that is caused by the absorption and scattering of light by small particles suspended in the air.
* Haze-removal is an important problem because it affects downstream tasks like object classification, crowd estimation etc.
* Early dehazing techniques require either multiple images or depth information to estimate the clear image.
* Single image dehazing methods attempt to dehaze a scene from a single image.

Methods such as [MSCNN (Ren 2016)][1] and [Yang 2017][2] both estimate the light transmission map (t) of the hazy image, which they utilise to calculate the dehazed image using a physical model for hazy image formation.

> hazy = original*trans + airlight*(1-trans)

where the transmission map is generated from the depth data and the scattering coefficient (beta) of the haze. 

> trans = exp(-beta*depth)

The scattering coefficient models the quality of the haze, like the size of the scattering particles.


### Dataset Creation

Our dataset creation process follows [Ren 2016][1] to generate a synthetic dataset of hazy images from indoor scenes using the [NYU Depth V1 dataset][3]. MSCNN uses 6000 samples from the unlabelled dataset. We restrict ourselves to the 2284 samples in the labelled dataset due to resource constraints.

* We first remove 40 samples for validation. We don't create a separate test set. In the future the unlabelled dataset can provide a larger test set.
* Each image is scaled to 240 x 320.
* For each image in the training set, we create 3 hazy images.
	* A single value of airlight is sampled from the range [0.7,1.0] for all 3 images.
	* 3 separate values of the scattering coefficient (beta) is sampled from the range [0.5,1.2]. The upper limit is less than MSCNN's choice of 1.5, to simplify the problem.
	* The transmission map and the hazy image are then generated using the physical model.

The code for dataset creation can be found in [this notebook](/src/generate_data.ipynb).

### Network and Training



### Proposed approach : Smoothness Loss



### Experiments

We had two experiments to evaluate the variation with the ``patch_size`` and ``scaling_coefficient`` hyperparameters. We ran both experiments twice - once with a regular update rule, and once with the gradient similarity based update rule, for a total of 12 runs. Additionally a base model without the auxiliary loss was trained. The rest of the parameters were kept the same as described in the [Network and Training](#network-and-training). All models were trained for 50 epochs.

#### Patch size

* ``scaling_coefficient`` was kept constant at 100.
* ``patch_size`` was set to 1,3,5


#### Scaling Coefficient

* ``scaling_coefficient`` was set to 10,100,1000.
* ``patch_size`` was kept constant to 1

### Conclusion



## References


[1]: https://link.springer.com/chapter/10.1007/978-3-319-46475-6_10 "Single Image Dehazing via Multi-scale Convolutional Neural Networks"
[2]: https://arxiv.org/abs/1710.00279 "Image Dehazing using Bilinear Composition Loss Function"
[3]: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v1.html "NYU Depth V1 dataset"