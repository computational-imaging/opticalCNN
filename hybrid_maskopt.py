import model
import layers.optics as optics
from layers.utils import *

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import numpy as np
import tensorflow as tf

from glob import glob

from datetime import datetime

class PhaseMaskModel(model.Model):
    def __init__(self, psf_file,
                 dim, wave_res,
                 wavelength,
                 n,  z_file=None, mask_file=None,
                 ckpt_path=None):

        self.dim = dim
        self.wave_resolution = wave_res
        self.wavelength = wavelength
        self.n = n
        self.r_NA = wave_res[0]/2
        self.psf_file = psf_file
        self.mask_file = mask_file
        self.z_file = z_file

        super(PhaseMaskModel, self).__init__(name='PhaseMask_ONN', ckpt_path=ckpt_path)

    def _build_graph(self, x_train, hm_reg_scale, hm_init_type='random_normal'):
        #with tf.device('/device:GPU:0'):
        sensordims = (self.dim,self.dim)
        # Start with input image
        input_img = x_train/tf.reduce_sum(x_train)
        input_img = tf.image.resize_nearest_neighbor(input_img, size=wave_res)
        tf.summary.image('input_image', input_img)

        doAmplitudeMask=False
        doZernike=False
        doFourier=False
        doBinaryMask=False
        z_modes=350
        freq_range=.8
        
        if doBinaryMask: # if additional amplitude mask on top of phase mask
            binary_mask = np.load(self.mask_file)
        else: 
            binary_mask = None
        
        output_fullres = optical_conv_layer(input_img, hm_reg_scale, self.r_NA, n=self.n, wavelength=self.wavelength,
                                            coherent=False, amplitude_mask=doAmplitudeMask, zernike=doZernike,
                                            fourier=doFourier, binarymask=doBinaryMask, n_modes=z_modes, 
                                            freq_range=freq_range, 
                                            binary_mask_np = binary_mask,
                                            zernike_file=self.z_file, name='maskopt')

        # Attach images to summary
        tf.summary.image('output_fullres', output_fullres)
        
        # output_img = optics.Sensor(input_is_intensities=False, resolution=sensordims)(output_img)
        output_img = tf.image.resize_nearest_neighbor(output_fullres, size=sensordims)
        
        return output_img

    def _get_data_loss(self, model_output, ground_truth):
        model_output = tf.cast(model_output, tf.float32)
        ground_truth = tf.cast(ground_truth, tf.float32)
        
        # model_output /= tf.reduce_max(model_output)
        ground_truth /= tf.reduce_sum(ground_truth)
        with tf.name_scope('data_loss'):
            optics.attach_img('model_output', model_output)
            optics.attach_img('ground_truth', ground_truth)
        loss = tf.reduce_mean(tf.abs(model_output - ground_truth))
        return loss

    def _get_training_queue(self, batch_size, num_threads=4):
        dim = self.dim
        
        file_list = tf.matching_files('/media/data/onn/cifar10padded/im_*.png')
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
        size_ratio = dim/tf.reduce_min(img_height_width)

        # Extract a glimpse from the image
        #offset_center = tf.random_uniform([1,2], minval=0.0 + size_ratio/2, maxval=1.0-size_ratio/2, dtype=tf.float32)
        offset_center = tf.random_uniform([1,2], minval=0, maxval=0, dtype=tf.float32)
        offset_center = offset_center * img_height_width

        image = tf.image.extract_glimpse(image, size=[dim,dim], offsets=offset_center, centered=True, normalized=False)
        image = tf.squeeze(image, 0)

        convolved_image = tf.expand_dims(image, 0)
        psf = tf.convert_to_tensor(np.load(self.psf_file), tf.float32)
        psf /= tf.reduce_sum(psf)
        optics.attach_img('gt_psf', tf.expand_dims(tf.expand_dims(tf.squeeze(psf), 0), -1))
        
        psf = tf.expand_dims(tf.expand_dims(tf.squeeze(psf), -1), -1)
        # psf = tf.transpose(psf, [1,2,0,3])
        
        pad = int(dim/2)
        convolved_image = tf.abs(optics.fft_conv2d(fftpad(convolved_image, pad), fftpad_psf(psf, pad), adjoint=False))
        convolved_image = fftunpad(convolved_image, pad)
        convolved_image = tf.squeeze(convolved_image,axis=0)
        convolved_image /= tf.reduce_sum(convolved_image)

        image_batch, convolved_img_batch = tf.train.batch([image, convolved_image],
                                                          shapes=[[dim,dim,1], [dim,dim,1]],
                                                          batch_size=batch_size,
                                                          num_threads=4,
                                                          capacity=4*batch_size)

        return image_batch, convolved_img_batch


if __name__=='__main__':
    tf.reset_default_graph()

    dim = 328
    scale = 1
    wave_res = np.array((scale*dim,scale*dim))
    wavelength = 532e-9
    n = 1.5090 # 1.4599
    num_steps = 20001
    
    psf_file = 'assets/psf_hybrid_optneg_8x9_1e-1.npy'

    phasemask = PhaseMaskModel(psf_file, dim, wave_res, wavelength, n, None, None, ckpt_path=None)

    now = datetime.now()
    runtime = now.strftime('%Y%m%d-%H%M%S')
    run_id = 'optneg_8x9_visual/' + runtime + '/'
    log_dir = os.path.join('checkpoints/hybrid_cifar10/', run_id)
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    
    phasemask.fit(model_params = {'hm_reg_scale':1e-1},
                   opt_type = 'ADAM',
                   #opt_params = {'beta1':0.8, 'beta2':0.999, 'epsilon':1.},
                   opt_params = {'momentum':0.5, 'use_nesterov':True},
                   decay_type = 'polynomial',
                   decay_params = {'decay_steps':num_steps, 'end_learning_rate':1e-9},
                   batch_size=1,
                   adadelta_learning_rate = 1, 
                   starter_learning_rate = 0.0005,
                   num_steps_until_save=2000,
                   num_steps_until_summary=200,
                   logdir = log_dir,
                   num_steps = num_steps)
