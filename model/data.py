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
    def __init__(self, folders_train, folders_val, num_classes):
        '''
        All paths to the images to train and validate.
        
        Inputs
        ---------
        folders_train : list
            A list of folders where are the dataset to train.
        folders_val : list
            A list of folders where are the dataset to validation.
        num_classes : int
            Number of classes
        '''
        
        self.num_classes = num_classes
        
        # map paths and split dataset
        self.path_train = []; self.path_test = []

        for path in folders_train:
            self.path_train.extend(glob(os.path.join(path,'*_image.jpg')))

        for path in folders_val:
            self.path_test.extend(glob(os.path.join(path,'*_image.jpg')))

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
        '''
        Return the number of data contained in each set
        
        Inputs
        ---------
        data : str
            Either train or test
            
        Returns
        ---------
        Size of the choosen dataset.
        '''
        
        assert data=='train' or data=='test', f'Data must be train or test, received {data}'
        if data=='train':
            return len(self.path_train)
        else:
            return len(self.path_test)

    def augment(self, image, mask):
        ''' Function for augmentation '''
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
            
            # open all masks based on the image path
            mask = np.zeros((*img.shape[:2], self.num_classes), dtype='uint8')
            path_mask = path.replace('image.jpg', 'mask')
            for pm in glob(path_mask+'*.jpg'):
                # get the class for the mask
                clas = int(pm.replace('.jpg', '').split('_')[-1])
                msk  = imread(pm, resize=resize)[...,0]
                
                # adjust mask
                msk = cv2.threshold(msk,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                mask[...,clas] = msk
            
            mask = SegmentationMapsOnImage(mask, shape=img.shape)
            
            # augmentation
            if augment:
                img, mask = self.augment(img, mask)

            mask = mask.get_arr()
            img = self.norm(img)
            
            # draw contours
            contours = np.zeros_like(mask)
            for i in range(mask.shape[-1]):
                base = np.zeros_like(img)
                cnts = cv2.findContours(mask[...,i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                contours[...,i] = cv2.drawContours(base, cnts, -1, (1,1,1), 2)[...,0]

            yield np.float32(img), {'mask_output':np.float32(mask),
                                    'contour_output':np.float32(contours)}
            
        return generator
            
    def flow(self, data, batch_size, resize=None):
        '''
        Tensorflow iterator
        
        Inputs
        ----------
        data : str
            Dataset to be used. Either train or test.
        batch_size : int
            Size of the batch.
        resize : tuple, default=None
            Value to resize images.
        '''
        
        # verify dataset to be used
        if tf.equal(data, 'train'):
            paths = self.path_train
            augment = True
        else:
            paths = self.path_test
            augment = False
        
        # index to each path sample
        indexes = [i for i in range(len(paths))]
        
        # get one image to verify shapes
        x, y = next(self.get_generator(paths, augment, resize)(0))
        
        # Tensorflow Dataset API options
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset
            dataset = dataset.from_tensor_slices(indexes)
            dataset = dataset.interleave(lambda index:tf.data.Dataset.from_generator(self.get_generator(paths, augment, resize),
                            (tf.float32, {'mask_output':tf.float32, 'contour_output':tf.float32}),
                            output_shapes=(x.shape, {'mask_output':y[0].shape, 'contour_output':y[1].shape}),
                            args=(index,)),
                        cycle_length=batch_size,
                        block_length=1,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat()
        
        return dataset
                    
    def view_data(self, data='train', batch_size=4, resize=None):
        '''
        Method to visualize the data returned by the generator (for verification purposes)
        
        Inputs
        ----------
        data : str
            Dataset to visualized. Either train or test.
        batch_size : int
            Size of the batch.
        resize : tuple, default=None
            Value to resize images.
        '''
        
        assert data=='train' or data=='test', f'Data must be train or test, received {data}'
        
        x, y = next(iter(self.flow(data, batch_size, resize)))
        x = x.numpy(); mask = y['mask_output'].numpy(); cnt = y['contour_output'].numpy()
        
        print('Batch X: ', x.shape, x.min(), x.max())
        print('Batch Mask: ', mask.shape, mask.min(), mask.max())
        print('Batch Contour: ', cnt.shape, cnt.min(), cnt.max())
        
        mask = [SegmentationMapsOnImage(m, shape=img.shape) for m in mask]
        cnt  = [SegmentationMapsOnImage(c, shape=img.shape) for c in cnt]
        
        plt.figure(figsize=(10,10))
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0]//2+1, 2, i+1)
            plt.imshow(np.hstack([
                                  self.denorm(x[i]),
                                  mask[i].draw_on_image(self.denorm(x[i])),
                                  cnt[i].draw_on_image(self.denorm(x[i]))
                                ]))
            
    def predict_data(self, model, data='test', batch_size=1, resize=None):
        '''
        Method to visualize the output of a trained model on train/test data
        
        Inputs
        ----------
        model: tf.keras.models.Model
            Model to be used.
        data : str
            Dataset to visualized. Either train or test.
        batch_size : int
            Size of the batch.
        resize : tuple, default=None
            Value to resize images.
        '''
        
        assert data=='train' or data=='test', f'Data must be train or test, received {data}'
        
        x, y = next(iter(self.flow(data, batch_size, resize)))
        x = x.numpy(); mask = y['mask_output'].numpy()
        
        pred = model.predict(x)[0]
        
        mask = [SegmentationMapsOnImage(m, shape=img.shape) for m in mask]
        pred = [SegmentationMapsOnImage(p, shape=img.shape) for p in pred]
        
        plt.figure(figsize=(10,10))
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0]//2+1, 2, i+1)
            plt.imshow(np.hstack([
                                  self.denorm(x[i]),
                                  mask[i].draw_on_image(self.denorm(x[i])),
                                  pred[i].draw_on_image(self.denorm(x[i]))
                                ]))
            plt.title('Input | Ground truth | Predicted')
    
    def predict_input(self, model, input_image):
        '''
        Method to visualize trained model on an input image
        
        Inputs
        ----------
        model: tf.keras.models.Model
            Model to be used.
        data : str
            Dataset to visualized. Either train or test.
        batch_size : int
            Size of the batch.
        resize : tuple, default=None
            Value to resize images.
        '''
        
        input_data = self.norm(input_image)
        p = model.predict(input_data[np.newaxis,...])[0][0,...]
        
        pred = SegmentationMapsOnImage(p, shape=input_image.shape)
        
        return p.draw_on_image(input_image)