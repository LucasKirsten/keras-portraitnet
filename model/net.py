import sys
sys.path.append('./')
from utils import *

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

class Model():
    
    def __init__(self,
              model_name,
              classes,
              input_shape   = (224,224,3),
              dropout       = 0.2,
              learning_rate = 1e-3,
              loss          = {'mask_output':'binary_crossentropy', 'contour_output':binary_focal_loss},
              loss_weights  = {'mask_output':1., 'contour_output':1.},
              metrics       = [jaccard_distance, 'acc']):
        
        model_name    = self.model_name
        classes       = self.classes
        input_shape   = self.input_shape
        dropout       = self.dropout
        learning_rate = self.learning_rate
        loss          = self.loss
        loss_weights  = self.loss_weights
        metrics       = self.metrics
        
    def build(self):
        
        def ConvBloc(input_tensor, n_filters, ksize, strides=1):
            u = L.DepthwiseConv2D(ksize, strides=strides, use_bias=False, padding='same') (input_tensor)
            u = L.BatchNormalization() (u)
            u = L.Activation('relu') (u)
            u = L.Conv2D(n_filters, 1, use_bias=False, padding='same', activation='relu') (u)
            u = L.BatchNormalization() (u)
            u = L.Activation('relu') (u)
            return u

        def DoubleConv2D(input_tensor, n_filters, ksize=3):
            u = L.SeparableConv2D(filters=n_filters, kernel_size=ksize, padding='same') (input_tensor)
            u = L.BatchNormalization() (u)
            u = L.Activation('relu') (u)
            u = L.SeparableConv2D(filters=n_filters, kernel_size=ksize, padding='same') (u)
            u = L.BatchNormalization() (u)
            u = L.Activation('relu') (u)
            return u

        def DimReduceBlock(input_tensor, n_filters, ksize, dropout):
            p = DoubleConv2D(input_tensor, n_filters, ksize)
            p = L.MaxPooling2D((2, 2)) (p)
            p = L.Dropout(dropout) (p)
            return p

        def DimExpanderBloc(input_tensor, concat_layer, n_filters, ksize, dropout):
            t = L.Conv2D(n_filters, 1, use_bias=False, padding='same') (input_tensor)
            t = L.BatchNormalization() (t)
            t = L.Activation('relu') (t)

            u = DoubleConv2D(input_tensor, n_filters, ksize)
            u = L.Add() ([t, u])

            u = L.Add() ([u, concat_layer])
            u = L.UpSampling2D() (u)
            u = L.Dropout(dropout)(u)

            return u

        input_tensor = L.Input(self.input_shape)
        
        # first convolution
        p = L.Conv2D(1, 1, use_bias=False, padding='same') (input_tensor)

        # encoder
        concat_layers = []
        for (f,k) in filters:
            p = DimReduceBlock(p, f, k, self.dropout)
            concat_layers.append(p)

        # decoder
        for layer, (f,k) in zip(concat_layers[::-1], filters[::-1]):
            p = DimExpanderBloc(p, layer, f, k, self.dropout)

        mask    = L.Conv2D(self.classes, (1, 1), use_bias=False, activation='sigmoid', name='mask_output') (p)
        contour = L.Conv2D(self.classes, (1, 1), use_bias=False, activation='sigmoid', name='contour_output') (p)

        model =  tf.keras.models.Model(input_tensor, [mask, contour], name=self.model_name)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss=self.loss,
                      loss_weights=self.loss_weights,
                      metrics=self.metrics)
        
        return model