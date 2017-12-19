import model
import layers.optics as optics

import numpy as np
import tensorflow as tf

from glob import glob

class CollimatorModel(model.Model):
    def __init__(self,
                 wave_resolution,
                 distance,
                 wave_length,
                 discretization_step,
                 refractive_index,
                 ckpt_path):

        self.wave_resolution = wave_resolution
        self.wave_length = wave_length
        self.distance = distance
        self.discretization_step = discretization_step

        super(CollimatorModel, self).__init__(name='Collimator_Test', ckpt_path=ckpt_path)

    def _build_graph(self, x_train, hm_reg_scale, hm_init_type='random_normal'):
        with tf.device('/device:GPU:2'):
            input_field = tf.ones([1,self.wave_resolution[0],self.wave_resolution[1],1])

            # Get the PSF
            #if hm_init_type=='random_normal':
            height_map_initializer=tf.random_uniform_initializer(minval=0.999e-4, maxval=1.001e-4)
            #print(height_map_initializer)
            field = optics.height_map_element(input_field,
                                              wave_lengths=self.wave_length,
                                              height_map_regularizer=optics.laplace_l1_regularizer(hm_reg_scale),
                                              height_map_initializer=height_map_initializer,
                                              name='height_map_optics')
            field = optics.circular_aperture(field)
            field = optics.propagate_fresnel(field,
                                             distance=self.distance,
                                             input_sample_interval=self.discretization_step,
                                             wave_lengths=self.wave_length)
            psf_graph = optics.Sensor(input_is_intensities=False, resolution=(512,512))(field)
            psf_graph /= tf.reduce_sum(psf_graph)
            psf_graph = tf.cast(psf_graph, tf.float32)
            tf.summary.image('predicted_psf', tf.log(psf_graph + 1e-9))
            psf_graph = tf.transpose(psf_graph, [1,2,0,3]) # (height, width, 1, 1)

            # Convolve with input img
            #x_train = tf.cast(x_train, tf.float32)
            #output_img = tf.nn.convolution(x_train, psf_graph, padding="SAME")
            output_img = optics.fft_conv2d(x_train, psf_graph)

            tf.summary.image('input_image', x_train)
            tf.summary.image('output_image', output_img)
            return output_img

    def _get_data_loss(self, model_output, ground_truth):
        model_output = tf.cast(model_output, tf.float32)
        ground_truth = tf.cast(ground_truth, tf.float32)
        loss = tf.reduce_mean(tf.abs(model_output - ground_truth))
        return loss

    def _get_training_queue(self, batch_size, num_threads=4):
        file_list = tf.matching_files('./test_imgs/*.png')
        filename_queue = tf.train.string_input_producer(file_list)

        image_reader = tf.WholeFileReader()

        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_png(image_file,
                                    channels=1,
                                    dtype=tf.uint8)
        image = tf.cast(image, tf.float32) # Shape [height, width, 1]
        image = tf.expand_dims(image, 0)
        image /= 255.

        # Get the ratio of the patch size to the smallest side of the image
        img_height_width = tf.cast(tf.shape(image)[1:3], tf.float32)
        size_ratio = 512. / tf.reduce_min(img_height_width)

        # Extract a glimpse from the image
        offset_center = tf.random_uniform([1,2], minval=0.0 + size_ratio/2, maxval=1.0-size_ratio/2, dtype=tf.float32)
        offset_center = offset_center * img_height_width

        image = tf.image.extract_glimpse(image, size=[512,512], offsets=offset_center, centered=False, normalized=False)
        image = tf.squeeze(image, 0)

        convolved_image = tf.expand_dims(image, 0)
        psf = tf.convert_to_tensor(np.load('test_imgs/cubic_phase_shifts_psf_fraunhofer.npy'), tf.float32)
        psf /= tf.reduce_sum(psf)
        psf = tf.transpose(psf, [1,2,0,3])
        tf.summary.image('gt_psf', tf.log(tf.expand_dims(tf.expand_dims(tf.squeeze(psf), 0), -1)+1e-9))
        convolved_image = optics.fft_conv2d(convolved_image, psf)
        convolved_image = tf.squeeze(convolved_image,axis=0)

        image_batch, convolved_img_batch = tf.train.batch([image, convolved_image],
                                                          shapes=[[512,512,1], [512,512,1]],
                                                          batch_size=batch_size,
                                                          num_threads=4,
                                                          capacity=4*batch_size)

        return image_batch, convolved_img_batch


if __name__=='__main__':
    tf.reset_default_graph()

    wave_resolution = np.array((512,512))
    wave_length = 532e-9
    discretization_step = 1e-6
    distance = 1e-2
    refractive_index=1.5
    num_steps = 5000

    collimator = CollimatorModel(wave_resolution,
                                 distance,
                                 wave_length,
                                 discretization_step,
                                 refractive_index,
                                 ckpt_path=None)

    collimator.fit(model_params = {'hm_reg_scale':1e-1},
                   opt_type = 'sgd_with_momentum',
                   #opt_params = {'beta1':0.8, 'beta2':0.999, 'epsilon':1.},
                   opt_params = {'momentum':0.5, 'use_nesterov':True},
                   decay_type = 'polynomial',
                   decay_params = {'decay_steps':num_steps, 'end_learning_rate':1e-9},
                   batch_size=1,
                   starter_learning_rate = 1e-2,
                   num_steps_until_save=500,
                   num_steps_until_summary=20,
                   logdir = '/media/data/checkpoints/flatcam/testing/',
                   num_steps = num_steps)
