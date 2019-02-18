# Automated annotation of cellular cryo-electron tomograms using convolutional neural network

The neural network architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) and based on the previous work [Convolutional neural networks for automated annotation of cellular cryo-electron tomograms](https://www.nature.com/articles/nmeth.4405)

---


### Model

![model.jpg](model.jpg)

Loss function:

![loss.jpg](loss.JPG)

### Training
The input are annotated tomograms, they are annotated by experienced researchers in the field with the help of either previous 4-layer shallow CNN or manually. Different features have corresponding labels.

Learning rate is 0.001, batch size is 4, epoch number is 50, Adam optimizer.

Evaluation metric: F1 score

The outputs will be 2D slices with each pixel annotated with one of the features. Merging can be done sequentially to get a 3D annotated tomogram or each feature can be stored in an individual togogram for further analysis.



---

## How to use
1. Prepare pre-processed data and label as described.
2. Use main.py for trainning and testing.
3. Use predict.py on new dataset.

### Dependencies

- [Pytorch](https://pytorch.org/)
