# Keras PortraitNet

A Keras implementation of the Portrait Net architecture. Original paper available at: https://www.yongliangyang.net/docs/mobilePotrait_c&g19.pdf

## Pre-requisites
- ```Python >= 3.6 ```
- ```Tensorflow 2.x```
- ```Imgaug```
- ```OpenCV```
- ```Jupyter Notebook```

## Expected dataset
It is expected that the dataset is contained in one or more folders. Each folder must contain the input image with suffix ```_image``` and all the segmentation masks with suffix ```_mask_<nr of class>``` (starts at 0). An example of folder struture can be:
```
- Dataset 1
-- img1_image.jpg
-- img1_mask_0.jpg
-- img1_mask_2.jpg
```
This shows an example where the ```img1``` has 2 segmentation masks with ids 0 and 2. The segmentation for the id class 1, as it is not provided, will be considere as a zero matrix.

**Important**: it is not expected the contour images, since the data generator script grab them using the find contours algorithm from OpenCV in the segmentation masks.

## Usage
You can start using this project with the provided jupyter notebook containing all aditional informations.

## Contributions
Contributions are allways welcome to improve this repository :D