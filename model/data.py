''' This script hold the class to handle the dataset (pre-processing, data loader, data visualization etc). You can modify the methods of this class as you wish as long as you keep their functionalities purposes. '''

import sys
sys.path.append('./')
from utils import *

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.parameters as iap

from glob import glob

class DataLoader(object):
    ''' Data loader class examplified when we have the data in memory '''
    def __init__(self, folders_train, folders_val):
        ''' All paths to the images to train and validate '''
        
        # map paths and split dataset
        self.path_train = []; self.path_test = []

        for path in folders_train:
            self.path_train.extend(glob(path+'/*_image.jpg'))

        for path in folders_val:
            self.path_test.extend(glob(path+'/*_image.jpg'))

        print(f'Total train: {len(self.path_train)}, Total val: {len(self.path_test)}')
        
        # options for augmentation
        self.aug = iaa.SomeOf((0,3), [
                    iaa.Affine(rotate=(-10, 10), scale={"x": (0.5, 1.2), "y": (0.5, 1.2)}),
                    iaa.AdditiveGaussianNoise(scale=0.2*255),
                    iaa.GaussianBlur(sigma=(0.0, 3.0)),
                    iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1.0),
                                               sigmoid_thresh=iap.Normal(10.0, 5.0)),
                    iaa.Add(50, per_channel=True),
                    iaa.WithChannels(0, iaa.Add((10, 100))),
                    iaa.Sharpen(alpha=0.2),
                    iaa.Fliplr(),
                    iaa.Flipud()
                ])
        
    def data_size(self, data):
        ''' Return the number of data contained in each set '''
        
        assert data=='train' or data=='test', f'Data must be train or test, received {data}'
        if data=='train':
            return len(self.path_train)
        else:
            return len(self.path_test)

    def augment(self, image, mask):
        ''' Function for augmentation, if neccessary '''
        return self.aug(image=image, segmentation_maps=mask)
        
    def norm(self, img):
        ''' Function to normalize the data '''
        return img/255.
    
    def denorm(self, img):
        ''' Function to de-normalize the data for visualization purposes '''
        return np.uint8(img*255)
    
    @tf.autograph.experimental.do_not_convert
    def get_generator(self, paths, augment, resize=None):
        ''' Generator of data '''
        
        def generator(index):
            path = paths[index]
            
            # open images
            img = imread(path, resize=resize)
            # open mask based on the image path
            mask = imread(path.replace('image', 'mask'), resize=resize)[...,0]

            # adjust mask
            _,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask = SegmentationMapsOnImage(np.uint8(mask/255.), shape=img.shape)

            # augmentation
            if augment:
                img, mask = self.augment(img, mask)

            mask = mask.get_arr()
            img = self.norm(img)

            # draw contour based on the mask
            contours = np.zeros_like(mask)
            cnts, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours = cv2.drawContours(contours, cnts, -1, (1,1,1), 2)

            # adjust channels
            contours = contours[...,np.newaxis]
            mask = mask[...,np.newaxis]

            yield np.float32(img), {'mask_output':np.float32(mask), 'contour_output':np.float32(contours)}
            
        return generator
            
    def flow(self, data, batch_size, resize=None):
        ''' Tensorflow iterator '''
        
        # verify dataset to be used
        if tf.equal(data, 'train'):
            paths = self.path_train
            augment = True
        else:
            paths = self.path_test
            augment = False
        
        # index to each path sample
        indexes = [i for i in range(len(paths))]
        
        # Tensorflow Dataset API options
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset
            dataset = dataset.from_tensor_slices(indexes)
            dataset = dataset.interleave(lambda index:tf.data.Dataset.from_generator(self.get_generator(paths, augment, resize),
                            (tf.float32, {'mask_output':tf.float32, 'contour_output':tf.float32}),
                            output_shapes=((224,224,3), {'mask_output':(224,224,1), 'contour_output':(224,224,1)}),
                            args=(index,)),
                        cycle_length=batch_size,
                        block_length=1,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat()
        
        return dataset
                    
    def view_data(self, data='train', batch_size=4, resize=None):
        ''' Method visualize the data returned by the generator (to verification purposes) '''
        
        assert data=='train' or data=='test', f'Data must be train or test, received {data}'
        
        x, y = next(iter(self.flow(data, batch_size, resize)))
        x = x.numpy(); y1 = y['mask_output'].numpy(); y2 = y['contour_output'].numpy()
        
        print('Batch X: ', x.shape, x.min(), x.max())
        print('Batch Y1: ', y1.shape, y1.min(), y1.max())
        print('Batch Y2: ', y2.shape, y2.min(), y2.max())
        
        plt.figure(figsize=(10,10))
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0]//2+1, 2, i+1)
            plt.imshow(np.hstack([self.denorm(x[i]),
                                  cv2.merge([self.denorm(y1[i][...,0])]*3),
                                  cv2.merge([self.denorm(y2[i][...,0])]*3)]))
            
    def predict_data(self, model, data='test', batch_size=1, resize=None):
        ''' Method to visualize trained model on train/test data '''
        
        assert data=='train' or data=='test', f'Data must be train or test, received {data}'
        
        x, y = next(iter(self.flow(data, batch_size, resize)))
        x = x.numpy(); y = y['mask_output'].numpy()
        p = model.predict(x)[0]
        
        plt.figure(figsize=(10,10))
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0]//2+1, 2, i+1)
            plt.imshow(np.hstack([self.denorm(x[i]), cv2.merge([self.denorm(y[i][...,0])]*3), cv2.merge([self.denorm(p[i][...,0])]*3)]))
            plt.title('Input | Ground truth | Predicted')
    
    def predict_input(self, model, input_image):
        ''' Method to visualize trained model on an input image '''
        
        input_image = self.norm(input_image)
        p = model(input_image[np.newaxis,...])[1].numpy()
        
        return cv2.merge([self.denorm(p[0,...,0])]*3)