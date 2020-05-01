## Project Overview

This project proposes a novel auxiliary loss for the task of transmission estimation in CNN based single image dehazing methods. This auxiliary loss models surface smoothness by aligning the normals within a small patch with the objective of removing surface artifacts from appearing on the transmission map, such as the one shown below as it hinders haze removal.

![Comparison with and without smoothness loss](/cover_example.png)

## Table of Contents

1. [Project Description](#project-description)
	1. [Dataset Creation](#dataset-creation)
	2. [Network and Training](#network-and-training)
	3. [Proposed Approach (Smoothness Loss)](#proposed-approach)
2. [Experiments](#experiments)
3. [Observations](#observations)
4. [Conclusion](#conclusion)
5. [Credits](#credits)
6. [References](#references)

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


### Proposed Approach

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

## Experiments

We had two experiments to evaluate the variation with the ``patch_size`` and ``scaling_coefficient`` hyperparameters. We ran both experiments twice - once with a regular update rule, and once with the gradient similarity based update rule, for a total of 12 runs. Additionally a base model without the auxiliary loss was trained. The rest of the parameters were kept the same as described in the [Network and Training](#network-and-training). All models were trained for 50 epochs.

* Qualitative evaluation : We pick 5 images at random from the test set and plot the transmission maps and dehazed images in a grid format.
* Quantative evaluation : We compute PSNR and SSIM metrics over the test set for both transmission maps and dehazed images.

Evaluation can be found in [src/evaluate_network.ipynb](https://nbviewer.jupyter.org/github/abarthakur/SI_dehazing/blob/master/src/evaluate_network.ipynb).

*Legend for tables*

* _orig : comparison of dehazed image with original clear image.
* _best : comparison of dehazed image with best case dehazing using the ground truth transmission.
* _tmap : comparison of predicted transmission map to ground truth.

### patch_size

* ``scaling_coefficient`` was kept constant at 100.
* ``patch_size`` was set to 1,3,5

*Usual Update*

| model_name   |   psnr_orig |   ssim_orig |   psnr_best |   ssim_best |   psnr_tmap |   ssim_tmap |
|:-------------|------------:|------------:|------------:|------------:|------------:|------------:|
| base         |     8.73467 |    0.526044 |    14.1001  |    0.656557 |     20.9011 |    0.724311 |
| patch_1      |     9.92582 |    0.604556 |    16.4187  |    0.746404 |     20.5392 |    0.71034  |
| patch_3      |     5.58509 |    0.317889 |     9.84629 |    0.410807 |     13.8452 |    0.18324  |
| patch_5      |     8.11264 |    0.495393 |    13.5808  |    0.630477 |     13.8452 |    0.18324  |

*Gradient Similarity Update*

| model_name   |   psnr_orig |   ssim_orig |   psnr_best |   ssim_best |   psnr_tmap |   ssim_tmap |
|:-------------|------------:|------------:|------------:|------------:|------------:|------------:|
| base         |     8.73467 |    0.526044 |     14.1001 |    0.656557 |     20.9011 |    0.724311 |
| patch_1      |     9.05775 |    0.550585 |     15.2169 |    0.671298 |     18.3263 |    0.673456 |
| patch_3      |     9.44283 |    0.583998 |     15.2239 |    0.693528 |     17.7086 |    0.66475  |
| patch_5      |     9.64712 |    0.595597 |     14.875  |    0.699366 |     17.3077 |    0.655023 |

### scaling_coefficient

* ``scaling_coefficient`` was set to 10,100,1000.
* ``patch_size`` was kept constant to 3

*Usual Update*

| model_name   |   psnr_orig |   ssim_orig |   psnr_best |   ssim_best |   psnr_tmap |   ssim_tmap |
|:-------------|------------:|------------:|------------:|------------:|------------:|------------:|
| base         |     8.73467 |    0.526044 |     14.1001 |    0.656557 |     20.9011 |    0.724311 |
| scale_10     |     8.63107 |    0.528289 |     15.0775 |    0.658703 |     18.5706 |    0.680066 |
| scale_100    |     6.0181  |    0.34102  |     10.8521 |    0.444228 |     13.8452 |    0.18324  |
| scale_1000   |     6.7128  |    0.364493 |     10.9262 |    0.46636  |     13.8452 |    0.18324  |

*Gradient Similarity Update*

| model_name   |   psnr_orig |   ssim_orig |   psnr_best |   ssim_best |   psnr_tmap |   ssim_tmap |
|:-------------|------------:|------------:|------------:|------------:|------------:|------------:|
| base         |     8.73467 |    0.526044 |     14.1001 |    0.656557 |     20.9011 |    0.724311 |
| scale_10     |     9.70443 |    0.583711 |     15.6644 |    0.697914 |     18.0569 |    0.673297 |
| scale_100    |     9.1483  |    0.567842 |     14.8621 |    0.683551 |     17.8488 |    0.665046 |
| scale_1000   |     9.26631 |    0.577071 |     14.8071 |    0.690229 |     17.5615 |    0.65869  |

## Observations

* With the usual update rule, for large values of ``patch_size``(3,5) the model collapsed to predicting a single value. This is consistent with the fact that a front facing plane optimizes this formulation of surface smoothness. ``patch_size``=1 yields better dehazing results than the base model, although the transmission map of the base model is closer to the ground truth. This is an interesting observation, and its not clear why this should happen. For large ``scaling_coefficient`` values (100,1000 for ``patch_size``=3) the models collapse - again explained by the dominance of the auxiliary loss.
* With the selective update rule, model collapse is avoided. Larger patch sizes seem to correspond with better dehazing. However the current framework makes it difficult to make more arguments in this direction as we noticed that for several values of hyperparams, the gradient updates include the smoothness loss only a few times per epoch.

## Conclusion

 * The fact that the front facing plane is a trivial solution to our loss, is similar to an argument made in [Barron, Malik 2013][4] against minimizing the norm of mean-curvature (a second order descriptor of surfaces in differential geometry) to ensure surface smoothness. Instead they chose to model it differently - by ensuring that the differences in MC belonged to a Gaussian Mixture Model first fitted on the differences in the training set. We did not pursue this method because of initial difficulties in debugging it and also we were hoping that with correct scaling we will be able to mitigate this issue. But this might be a good direction for future work.
 * We used a gradient similarity based update rule to mitigate the issue of model collapse as observed in experiments. This had mixed success, as for larger values of patch_size we noticed that very few updates per epoch actually included the smoothness gradient - although enough to obviously make a difference as evidenced in the experiments.
 * Qualititave examination of the predicted transmission maps do show that the smoothness loss does "smoothen" the transmission map - however there is not much success in removing actual surface features. Alternative formulations of the smoothness loss should be explored as a first step in solving this problem.
* **Overall we conclude that current results are satisfactory enough to merit further work in this direction.**

## Credits

* This project and the basic proposed approach of using a smoothness prior with MSCNN architecture was initially conceived as part of a group project completed for Computer Vision using Machine Learning taught by Dr Arijit Sur at IIT Guwahati in 2017.
* The group members for that project are - Aneesh Barthakur, Prateek Vij, Hritik Jain, Darshit Patel
* **However the current formulation of smoothness loss, usage of gradient similarity based update, the current codebase and experiments were created by Aneesh Barthakur (me) independently at a later time.** The initial codebase was in tensorflow, and this one is in pytorch.

## References

1. Single Image Dehazing via Multi-scale Convolutional Neural Networks : Ren et al 2018[link][1]
2. Image Dehazing using Bilinear Composition Loss Function : Yang et al 2017 [link][2]
3. NYU Depth V1 dataset : Silberman, Fergus 2011 [link][3]
4. Shape, Illumination, and Reflectance from Shading : Barron, Malik 2013 [link][4]
5. Adapting Auxiliary Losses Using Gradient Similarity : Du et al 2018 [link][5]


[1]: https://link.springer.com/chapter/10.1007/978-3-319-46475-6_10 "Single Image Dehazing via Multi-scale Convolutional Neural Networks"
[2]: https://arxiv.org/abs/1710.00279 "Image Dehazing using Bilinear Composition Loss Function"
[3]: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v1.html "NYU Depth V1 dataset"
[4]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2013/EECS-2013-117.pdf "Shape, Illumination, and Reflectance from Shading"
[5]: https://arxiv.org/pdf/1812.02224.pdf "Adapting Auxiliary Losses Using Gradient Similarity"
