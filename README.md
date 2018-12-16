# Faster Automated annotation of cellular cryo-electron tomograms using convolutional neural network

The neural network architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) and based on the previous work [Convolutional neural networks for automated annotation of cellular cryo-electron tomograms](https://www.nature.com/articles/nmeth.4405)

---

## Overview

### Data

The training dataset is TEM image stacks of [PC12 cell](https://www.ebi.ac.uk/pdbe/entry/emdb/EMD-8594), a tomogram after binning 4 is 864\*864\*94 pixels, 28 angstrom per pixel. There are four features to annotate, microtubules, ribosomes, single membranes and double membranes. 
Eventually the user will define the number of features, the input will be 3D reconstructed tomograms.

### Model

![img/u-net-architecture.png](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it easy to experiment with different interesting architectures depends on complexity of annotation task.

### Training
The input are annotated tomograms, they are annotated by experienced researchers in the field with the help of either previous 4-layer shallow CNN or manually. Different features have corresponding labels.
The outputs will be 2D slices with each pixel annotated with one of the features. Merging can be done sequentially to get a 3D annotated tomogram or each feature can be stored in an individual togogram for further analysis.
Loss function for the training is cross-entropy.


---

## How to use

The full U-Net will be implemented in the extensive cryo-EM software package [EMAN2](https://blake.bcm.edu/emanwiki/EMAN2), as one of the mode. And you can have an idea of how the previous 4-layer CNN works [here](https://blake.bcm.edu/emanwiki/EMAN2/Programs/tomoseg).

### Dependencies

 - EMAN2
 - keras >= 1.0
 
Also, this code should be compatible with Python version 2.7.14.
