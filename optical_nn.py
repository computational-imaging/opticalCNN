import model_classification
import layers.optics as optics
from layers.utils import * 

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import tensorflow as tf

from glob import glob

import itertools
from datetime import datetime

from tensorflow.examples.tutorials.mnist import input_data

class OpticalClassifier(model_classification.ClassificationModel):
    def __init__(self,
                 wavelength,
                 n,
                 ckpt_path):

        self.wave_resolution = (100,100)
        self.wavelength = wavelength
        self.n = n
        self.r_NA = 80

        super(OpticalClassifier, self).__init__(name='Optical_NN', ckpt_path=ckpt_path)

    def _optical_conv_layer(self, input_field, hm_reg_scale, activation=None, coherent=False,
                            name='optical_conv'):
        
        with tf.variable_scope(name):
            sensordims = self.wave_resolution
            input_field = tf.cast(input_field, tf.complex128)
            
            # Zero-centered fft2 of input field
            field = optics.fftshift2d_tf(optics.transp_fft2d(optics.ifftshift2d_tf(input_field)))

            # Build a phase mask, zero-centered
            height_map_initializer=tf.random_uniform_initializer(minval=0.999e-4, maxval=1.001e-4)
            # height_map_initializer=None
            pm = optics.height_map_element([1,self.wave_resolution[0],self.wave_resolution[1],1],
                                             wave_lengths=self.wavelength,
                                             height_map_initializer=height_map_initializer,
                                             name='phase_mask_height',
                                             refractive_index=self.n)
            # height_map_regularizer=optics.laplace_l1_regularizer(hm_reg_scale),
            
            # Get ATF and PSF
            otf = tf.ones([1,self.wave_resolution[0],self.wave_resolution[1],1])
            otf = optics.circular_aperture(otf, max_val = self.r_NA)
            otf = pm(otf)
            psf = optics.fftshift2d_tf(optics.transp_ifft2d(optics.ifftshift2d_tf(otf)))
            psf = optics.Sensor(input_is_intensities=False, resolution=sensordims)(psf)
            psf /= tf.reduce_sum(psf) # conservation of energy
            psf = tf.cast(psf, tf.float32)
            optics.attach_img('psf', psf)
            
            # Get the output image
            if coherent:
                field = optics.circular_aperture(field, max_val = self.r_NA)
                field = pm(field)
                tf.summary.image('field', tf.square(tf.abs(field)))
                output_img = optics.fftshift2d_tf(optics.transp_ifft2d(optics.ifftshift2d_tf(field)))
            else:
                psf = tf.expand_dims(tf.expand_dims(tf.squeeze(psf), -1), -1)
                output_img = tf.abs(optics.fft_conv2d(input_field, psf))
    
            # Apply nonlinear activation
            if activation is not None:
                output_img = activation(output_img)
                
            return output_img


    def _build_graph(self, x_train, hm_reg_scale):
        input_img = tf.reshape(x_train, [-1, 28, 28, 1])
        padamt = 20
        paddings = tf.constant([[0, 0,], [padamt, padamt], [padamt, padamt], [0, 0]])
        input_img = tf.pad(input_img, paddings)
        
        # input_shape = list(input_img.shape)
        # Resize the input image to bring it up to the wave resolution
        input_img = tf.image.resize_nearest_neighbor(input_img,
                                                    size=self.wave_resolution)

        tf.summary.image("input_img", input_img)
        tf.summary.scalar("input_img_max", tf.reduce_max(input_img))
        tf.summary.scalar("input_img_min", tf.reduce_min(input_img))
        

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        

        with tf.device('/device:GPU:2'):
            # h_conv1 = self._optical_conv_layer(input_img, hm_reg_scale=hm_reg_scale, name='opt_conv_1')
            # h_conv2 = self._optical_conv_layer(h_conv1, hm_reg_scale=hm_reg_scale, coherent=True, name='opt_conv_2')
            # h_conv3 = self._optical_conv_layer(h_conv2, hm_reg_scale=hm_reg_scale, coherent=True, name='opt_conv_3')

            # intensities = optics.Sensor(resolution=self.wave_resolution, input_is_intensities=False)(h_conv2)
            # tf.summary.image("Sensor readings", intensities)

            # spatial_coords = np.meshgrid([300, 600], np.linspace(100, 900, num=5).tolist())
            # spatial_coords = spatial_coords.transpose(1,2,0)[:,:,:,None]
            # spatial_coords = np.tile(spatial_coords, [input_shape[0], 1, 1, 1])
            
            # batch_coords = np.arange(input_shape[0])[:,None, None, None]
            # batch_coords = np.tile(batch_coords, [1, 2, 5, 1])
            
            # gather_nd_coords = np.concatenate([batch_coords, spatial_coords], axis=3)
            # logits = tf.gather_nd(intensities, gather_nd_coords)
            # logits = tf.reshape(logits, [input_shape[0], 10]) # 10 digit classes
            
      
            W_conv1 = weight_variable([36, 36, 1, 1])            
            h_conv1 = conv2d(input_img, tf.square(W_conv1))
            tf.summary.image("h_conv1", h_conv1)

            W_conv2 = weight_variable([36, 36, 1, 1])
            h_conv2 = conv2d(h_conv1, tf.square(W_conv2))
            tf.summary.image("h_conv2", h_conv2)
            
           
            fc_size = self.wave_resolution[0]*self.wave_resolution[1]
            h_conv2_flat = tf.reshape(h_conv2, [-1, fc_size])
            W_fc1 = weight_variable([fc_size, 10])           
            logits = tf.matmul(h_conv2_flat, (W_fc1))
        
        
            # sum up separate parts of image to get logits
            #intensities_split = tf.split(h_conv3, num_or_size_splits=10, axis=1)
            # logits = tf.transpose(tf.reduce_mean(intensities_split, axis=[2,3,4]))
            
                                    
            predicted_class = tf.argmax(logits, axis=1)
            # tf.summary.tensor_summary('predicted_class', predicted_class) # how to see this?

            return logits

    def _get_data_loss(self, model_output, ground_truth):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model_output, labels=ground_truth)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)
        
        correct_prediction = tf.equal(tf.argmax(model_output, 1), tf.argmax(ground_truth, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('training accuracy', accuracy)
        
        return loss, accuracy

    def _get_training_queue(self, batch_size):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        xs, ys = mnist.train.next_batch(1000)
        
        image_batch, label_batch = tf.train.batch([xs, ys],
                                                          batch_size=batch_size,
                                                          num_threads=4,
                                                          capacity=4*batch_size,
                                                          enqueue_many=True) #shapes=[[100, 784], [100, 10]],
        return tf.squeeze(image_batch), tf.squeeze(label_batch)
    
    def _get_validation_queue(self):
        mnist = input_data.read_data_sets('MNIST_data/", one_hot=True)
        image_batch, label_batch = mnist.test.images, mnist.test.labels
        return image_batch, label_batch


if __name__=='__main__':
    tf.reset_default_graph()

    ckpt_path = None
    num_steps = 20000
    wavelength = 532e-9 
    n = 1.48

    optical_nn = OpticalClassifier(wavelength, n, ckpt_path)

    now = datetime.now()
    optical_nn.fit(model_params = {'hm_reg_scale':1e-2},
                   opt_type = 'sgd_with_momentum',
                   opt_params = {'momentum':0.9, 'use_nesterov':True},
                   decay_type = 'polynomial',
                   decay_params = {'decay_steps':num_steps, 'end_learning_rate':1e-9},
                   batch_size=50,
                   starter_learning_rate = 1e-3,
                   num_steps_until_save=1000,
                   num_steps_until_summary=200,
                   logdir = os.path.join('checkpoints/onn/', now.strftime('%Y%m%d-%H%M%S/')),
                   num_steps = num_steps)
