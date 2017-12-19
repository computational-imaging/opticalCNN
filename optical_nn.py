import model
import layers.optics as optics

import numpy as np
import tensorflow as tf

from glob import glob
import nn_architectures

import itertools

from tensorflow.examples.tutorials.mnist import input_data

class OpticalClassifier(model.Model):
    def __init__(self,
                 ckpt_path):

        self.wave_resolution = (1024,1024)

        super(OpticalClassifier, self).__init__(name='Optical_NN', ckpt_path=ckpt_path)

    def _optical_conv_layer(self,
                            input_field,
                            hm_reg_scale,
                            nonlinearity=True,
                            name='optical_conv'):
        with tf.variable_scope(name):
            input_field = tf.cast(input_field, tf.complex128)
            height_map_initializer=tf.random_uniform_initializer(minval=0.999e-4, maxval=1.001e-4)
            field = optics.propagate_fraunhofer_lens(field)
            field = optics.height_map_element(input_field,
                                              wave_lengths=self.wave_lengths,
                                              height_map_regularizer=optics.laplace_l1_regularizer(hm_reg_scale),
                                              height_map_initializer=height_map_initializer,
                                              name='height_map_optics')
            field = optics.circular_aperture(field)
            if nonlinearity:
                field = optics.propagate_fraunhofer_lens(field)
                field = optics.bacteriorhodopsin_nonlinearity(field)
            return field


    def _build_graph(self, x_train, hm_reg_scale):
        input_img = x_train
        input_shape = input_img.shape.as_list()
        # Resize the input image to bring it up to the wave resolution
        input_img = tf.image.resize_nearest_neighbor(input_img,
                                                     size=self.wave_resolution)

        tf.summary.image("input_img", input_img)
        tf.summary.scalar("input_img_max", tf.reduce_max(input_img))
        tf.summary.scalar("input_img_min", tf.reduce_min(input_img))

        with tf.device('/device:GPU:2'):
            conv_1 = self._optical_conv_layer(input_img, hm_reg_scale=hm_reg_scale, name='opt_conv_1')
            conv_2 = self._optical_conv_layer(conv_2, hm_reg_scale=hm_reg_scale, name='opt_conv_2')
            conv_3 = self._optical_conv_layer(conv_3, hm_reg_scale=hm_reg_scale, name='opt_conv_3')

            intensities = optics.Sensor(resolution=wave_resolution, input_is_intensities=False)(conv_3)
            tf.summary.image("Sensor readings", intensities)

            spatial_coords = np.meshgrid([300, 600], np.linspace(100, 900, num=5).tolist())
            spatial_coords = spatial_coords.transpose(1,2,0)[:,:,:,None]
            spatial_coords = np.tile(spatial_coords, [input_shape[0], 1, 1, 1])
            print(spatial_coords.get_shape())

            batch_coords = np.arange(input_shape[0])[:,None, None, None]
            batch_coords = np.tile(batch_coords, [1, 2, 5, 1])
            print(batch_coords.get_shape())

            gather_nd_coords = np.concatenate([batch_coords, spatial_coords], axis=3)

            logits = tf.gather_nd(intensities, gather_nd_coords)
            logits = tf.reshape(logits, [input_shape[0], -1])
            print(logits.get_shape())

            predicted_class = tf.argmax(logits)
            tf.summary.tensor_summary('predicted_class', predicted_class)

            return logits

    def _get_data_loss(self, model_output, ground_truth):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model_output, labels=ground_truth)

        loss = tf.reduce_mean(cross_entropy)
        return loss

    def _get_training_queue(self, batch_size):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        image_batch, label_batch = mnist.train.next_batch(batch_size)

        return image_batch, label_batch


if __name__=='__main__':
    tf.reset_default_graph()

    ckpt_path = None
    num_steps = 100000

    optical_nn = OpticalClassifier(ckpt_path)

    optical_nn.fit(model_params = {'hm_reg_scale':1e-1},
                   opt_type = 'sgd_with_momentum',
                   opt_params = {'momentum':0.9, 'use_nesterov':True},
                   decay_type = 'polynomial',
                   decay_params = {'decay_steps':num_steps, 'end_learning_rate':1e-9},
                   batch_size=20,
                   starter_learning_rate = 1e-2,
                   num_steps_until_save=5000,
                   num_steps_until_summary=200,
                   logdir = '/media/data/checkpoints/flatcam/testing/',
                   num_steps = num_steps)
