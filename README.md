# Automated annotation of cellular cryo-electron tomograms using convolutional neural network

The neural network architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) and based on the previous work [Convolutional neural networks for automated annotation of cellular cryo-electron tomograms](https://www.nature.com/articles/nmeth.4405)

---

## Overview

### Data
The	 raw	 data	 comes	 from	 three	 sources.	 The	 first	 source	 is	 a PC-12	 cell	tomogram	in	size	of	(96,	864,	868)	used	in	Muyuan	Chen’s	paper. This	data	is	acquired	at	low	magnification	on	cryo-electron	microscopy.	The	second	part	of	the	data	consists	of thirty-eight	neuron	cell	tomograms	acquired	at	medium	magnification	of	size	(n,	960,	960)	where	n	is	between	113	and	630 and with	median	 281.	 And	 the	 third	 part	 includes	 four high-magnification	 ribosome	tomograms	 from	EMPIAR	10064	in	size	of	(256,	1024,1024). Of	 these	 three	sets	of	data,	only	the	first	dataset	has	been	manually	annotated.Lots	of	efforts	are	devoted	in	data	preprocessing.	First,	top	andbottom	slices	from	 neuron	 and	 ribosome	 tomograms	 are	 excluded	 since	 these	 images	mostly	have	nothing	but	noise.	Then	38	neuron	cell	tomograms	are	combined	into	8	large	tomograms	in	order	to	obtain more	accurate	labelling.	All	neuron	cell	 and	 ribosome	 tomograms	 are	 semi-automatically	 labelled using	e2tomoseg_convnet.py	 developed	 by	Muyuan	 Chen.	 The	 annotated	 featuresincludes	 microtubule,	 ribosome,	 double	 layer	 membrane,	 single	 layermembrane	and	carbon	edge.	False	positives	from	semi-automatic	annotation are	largely	excluded	by	thresholding	and	manual	cleaning.	Mask	for	noise	and	five	 features	 are	 further	 encoded	with	 0	 <=	 number	 <	 6(class	 number). All	cleaned	3D	tomograms	and	their	corresponding	3D	masks	are	extracted	into	2D	images	with	each	image	being	cropped	into	four	images	of	size	512	by	512.The	 final	 dataset	 contains	 16,856	 512x512	 images.	 Dataset	 is	 randomlyshuffled	and	divided	into	train	set,	dev	set	and	test	set(80/10/10).

### Model

![img/u-net-architecture.png](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

Loss function:


### Training
The input are annotated tomograms, they are annotated by experienced researchers in the field with the help of either previous 4-layer shallow CNN or manually. Different features have corresponding labels.
The outputs will be 2D slices with each pixel annotated with one of the features. Merging can be done sequentially to get a 3D annotated tomogram or each feature can be stored in an individual togogram for further analysis.
Loss function for the training is cross-entropy.


---

## How to use



### Dependencies

- Pytorch
 
