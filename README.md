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

We propose a novel auxiliary loss to model surface smoothness, given by the mean of the cosine distance of the normals in a small patch. This idea was inspired by the paper [Shape, Illumination, and Reflectance from Shading][4] which proposed several priors to extract depth/shape of an object from a single image. 

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

* We use the MSCNN Coarse Network architecture. This takes 3-channel hazy_image and outputs the 1-channel transmission_map.
* The hazy image is fed to 3 blocks composed as ( convolution -> ReLu -> MaxPool -> Upsample ).
* Note that after each block, height and width remain same as input image.
* The last block is followed by a weighted linear combination of the last block's outputs.
* Then sigmoid activation is applied to give values in \[0,1\]

The configurations of the convolutional filters are 
1. 5x11x11 
2. 5x9x9
3. 10x7x7

where the format is (OUT_CHANNELSxFILTER_WIDTHxFILTER_HEIGHT).

* The optimizer used is Stochastic Gradient Descent with momentum of 0.9. (Same as MSCNN)
* L2 Weight decay is used with a coefficient of 5e-04. (Same as MSCNN)
* Batch size of 100 images is used (Same as MSCNN)
* A constant learning rate of 0.01 is used
	* Note that MSCNN used an LR schedule starting with 0.001 and decaying by 0.1 every 20 epochs (models were trained for 70 epochs). However we chose to use a constant learning rate to make the training process simpler.


### Proposed approach : Smoothness Loss

We propose a novel auxiliary loss to model surface smoothness, given by the mean of the cosine distance of the normals in a small patch.

>  1/(P\*P\*H\*W) <span style="font-size:20px;"> &sum; <sub>&forall; &#8741;(I1,J1)-(I2,J2)&#8741; &le; P </sub> (1 - N<sub>I1,J1</sub> . N<sub>I2,J2</sub>)</span>

where P is patch size, and H,W are the height and width of the image. The normals are calculated as 

> N = (T<sub>x</sub>/B, T<sub>y</sub>/B, 1/B ) where B = sqrt(1+T<sub>x</sub>^2 + T<sub>y</sub>^2) 

where T is the transmission map. T<sub>x</sub> and T<sub>y</sub> are the first order derivatives of T, which are calculated by convolving the [Sobel operators](https://en.wikipedia.org/wiki/Sobel_operator) with T. 

This loss is scaled and added to the Mean Squared Error loss used to train MSCNN. 

```
Total_loss = MSE_Loss(T_true,T_pred) + scaling_coefficient * Smoothness_Loss(T_pred, patch_size)
```

**Thus 2 new hyperparameters are introduced**, ``patch_size`` and ``scaling_coefficient``.

#### Gradient Similarity based update

* Although auxiliary losses can be useful to embed desirable biases in the training process, it is not clear when they help in the original task and when they do not.
* [Due et al 2018][5] propose to use the cosine similarity between gradients to determine if the gradient from an auxiliary loss should be used to update weights.
* More specifically, if ``grad_aux * grad_main > 0``, then we update the weights using ``grad_aux + grad_main`` as the gradient in the stochastic gradient update rule.
* We implement this idea and evaluate its efficacy.

### Experiments

We had two experiments to evaluate the variation with the ``patch_size`` and ``scaling_coefficient`` hyperparameters. We ran both experiments twice - once with a regular update rule, and once with the gradient similarity based update rule, for a total of 12 runs. Additionally a base model without the auxiliary loss was trained. The rest of the parameters were kept the same as described in the [Network and Training](#network-and-training). All models were trained for 50 epochs.

* Qualitative evaluation : We pick 5 images at random from the test set and plot the transmission maps and dehazed images in a grid format.
* Quantative evaluation : We compute PSNR and SSIM metrics over the test set for both transmission maps and dehazed images.

Evaluation can be found in [src/evaluate_network.ipynb](https://nbviewer.jupyter.org/github/abarthakur/SI_dehazing/blob/master/src/evaluate_network.ipynb).

#### Patch size

* ``scaling_coefficient`` was kept constant at 100.
* ``patch_size`` was set to 1,3,5

#### Scaling Coefficient

* ``scaling_coefficient`` was set to 10,100,1000.
* ``patch_size`` was kept constant to 1


## References

1. Single Image Dehazing via Multi-scale Convolutional Neural Networks : [link][1]
2. Image Dehazing using Bilinear Composition Loss Function : [link][2]
3. NYU Depth V1 dataset : [link][3]
4. Shape, Illumination, and Reflectance from Shading : [link][4]
5. Adapting Auxiliary Losses Using Gradient Similarity : [link][5]


[1]: https://link.springer.com/chapter/10.1007/978-3-319-46475-6_10 "Single Image Dehazing via Multi-scale Convolutional Neural Networks"
[2]: https://arxiv.org/abs/1710.00279 "Image Dehazing using Bilinear Composition Loss Function"
[3]: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v1.html "NYU Depth V1 dataset"
[4]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2013/EECS-2013-117.pdf "Shape, Illumination, and Reflectance from Shading"
[5]: https://arxiv.org/pdf/1812.02224.pdf "Adapting Auxiliary Losses Using Gradient Similarity"