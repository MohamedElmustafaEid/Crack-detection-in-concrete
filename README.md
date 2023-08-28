# Crack_Detection

This respository contains several functions for the detection of cracks in concrete.

The code is run from the jupyter notebook (crack_localization.ipynb), the work is divided into the following steps:

1- The dataset (retrieved from https://data.mendeley.com/datasets/5y9wdsg2zt/1) is split into three subsets: Training, validation and testing. And labeled into crack and non/crack image patches.

2- A convolutional Neural Network is trained with a final accuracy of 97.7% and a validation accuracy of 97.57%.

3- A testing image is loaded into the workspace, this image is split into n patches of size (227 * 227). The image patches will also be labeled crack/non-crack and saved in a dataframe alongside the patches names and their raw predicition (probability of class in numbers).

4- The classification dataframe is used to put marks on the crack patches before the image is reconstructed into its original shape.

5- The neural network and image reconstruction parts are still a work in progress.
