import abc

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import ifftshift
import fractions

import layers.optics as optics

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape, name=None):
  """Create a weight variable with appropriate initialization."""
  # initial = tf.truncated_normal(shape, stddev=0.1)
  # return tf.Variable(initial, name=name) 
  return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
  

def bias_variable(shape, name=None):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def shifted_relu(x):
    shift = tf.reduce_mean(x)
    return tf.nn.relu(x - shift) + shift

def cycle_W_conv(W_conv, din):
    # cycle through conv kernels
    w_list = []
    for i in range(din):
        w_list.append(tf.concat([W_conv[:,:,:,i:], W_conv[:,:,:,:i]], axis=3))
    return tf.concat(w_list, axis=2)

def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        
def fftpad(x, padamt):
    """Add padding before convolution with FFT"""
    paddings = tf.constant([[0, 0,], [padamt, padamt], [padamt, padamt], [0, 0]])
    return tf.pad(x, paddings)

def fftpad_psf(x, padamt):
    """Add padding before convolution with FFT""" 
    paddings = tf.constant([[padamt, padamt,], [padamt, padamt], [0, 0], [0, 0]]) #[x, y, 1, 1]
    return tf.pad(x, paddings)

def fftunpad(x, unpadamt):
    """Remove padding after convolution with FFT"""
    return x[:,unpadamt:-unpadamt, unpadamt:-unpadamt, :]

##############################
# NN layers
##############################

def optical_conv_layer(input_field, hm_reg_scale, r_NA, n=1.48, wavelength=532e-9, activation=None, 
                       coherent=False, amplitude_mask=False, zernike=False, fourier=False, binarymask=False,
                       n_modes = 1024, freq_range=.5, initializer=None, zernike_file='zernike_volume_256.npy',
                       binary_mask_np=None, name='optical_conv'):
    
    dims = input_field.get_shape().as_list()
    with tf.variable_scope(name):
        
        if initializer is None:
            initializer = tf.random_uniform_initializer(minval=0.999e-4, maxval=1.001e-4)
        
        if amplitude_mask:
            # Build an amplitude mask, zero-centered
            # amplitude_map_initializer=tf.random_uniform_initializer(minval=0.1e-3, maxval=.1e-2)
            mask = optics.amplitude_map_element([1,dims[1],dims[2],1], r_NA, 
                                             amplitude_map_initializer=initializer,
                                             name='amplitude_mask')
        
        else:
            # Build a phase mask, zero-centered
            if zernike:
                zernike_modes = np.load(zernike_file)[:n_modes,:,:]
                zernike_modes = tf.expand_dims(zernike_modes, -1)
                zernike_modes = tf.image.resize_images(zernike_modes, size=(dims[1], dims[2]))
                zernike_modes = tf.squeeze(zernike_modes, -1)
                mask = optics.zernike_element(zernike_modes, 'zernike_element', wavelength, n, r_NA, 
                                             zernike_initializer=initializer)
            elif fourier:
                mask = optics.fourier_element([1,dims[1],dims[2],1], 'fourier_element', 
                                       wave_lengths = wavelength, refractive_index=n,
                                       frequency_range = freq_range,
                                       height_map_regularizer=None,
                                       height_tolerance=None, # Default height tolerance is 2 nm.
                                       )
                                             
            else:
                # height_map_initializer=None
                mask = optics.height_map_element([1,dims[1],dims[2],1],
                                                 wave_lengths=wavelength,
                                                 height_map_initializer=initializer,
                                                 #height_map_regularizer=optics.laplace_l1_regularizer(hm_reg_scale),
                                                 name='phase_mask_height',
                                                 refractive_index=n)

        # Get ATF and PSF
        atf = tf.ones([1,dims[1],dims[2],1])
        #zernike=True
        if zernike:
            atf = optics.circular_aperture(atf, max_val = r_NA)
        atf = mask(atf)
        
        # apply any additional binary amplitude mask [1, dim, dim, 1]
        if binarymask:
            binary_mask = tf.convert_to_tensor(binary_mask_np, dtype=tf.float32)
            binary_mask = tf.expand_dims(tf.expand_dims(binary_mask, 0), -1)
            # optics.attach_img('binary_mask', binary_mask)
            atf = atf*tf.cast(binary_mask, tf.complex128)
            optics.attach_img('atf', tf.abs(binary_mask))
        
        # psf = optics.fftshift2d_tf(optics.transp_ifft2d(optics.ifftshift2d_tf(atf)))
        psfc = optics.fftshift2d_tf(optics.transp_ifft2d(optics.ifftshift2d_tf(atf)))
        psf = optics.Sensor(input_is_intensities=False, resolution=(dims[1],dims[2]))(psfc)
        psf /= tf.reduce_sum(psf) # conservation of energy
        psf = tf.cast(psf, tf.float32)
        optics.attach_summaries('psf', psf, True, True)

        # Get the output image
        if coherent:
            input_field = tf.cast(input_field, tf.complex128)
            # Zero-centered fft2 of input field
            field = optics.fftshift2d_tf(optics.transp_fft2d(optics.ifftshift2d_tf(input_field)))
            # field = optics.circular_aperture(field, max_val = r_NA)
            field = atf*field
            tf.summary.image('field', tf.log(tf.square(tf.abs(field))))
            output_img = optics.fftshift2d_tf(optics.transp_ifft2d(optics.ifftshift2d_tf(field)))
            output_img = optics.Sensor(input_is_intensities=False, resolution=(dims[1],dims[2]))(output_img)
            # does this need padding as well?
            
            # psfc = tf.expand_dims(tf.expand_dims(tf.squeeze(psfc), -1), -1)
            # padamt = int(dims[1]/2)
            # output_img = optics.fft_conv2d(fftpad(input_field, padamt), fftpad_psf(psfc, padamt), adjoint=False)
            # output_img = fftunpad(output_img, padamt)
            # output_img = optics.Sensor(input_is_intensities=False, resolution=(dims[1],dims[2]))(output_img)       
            
        else:
            psf = tf.expand_dims(tf.expand_dims(tf.squeeze(psf), -1), -1)
            # psf_flip = tf.reverse(psf,[0,1])
            # output_img = conv2d(input_field, psf)
            
            padamt = int(dims[1]/2)
            output_img = tf.abs(optics.fft_conv2d(fftpad(input_field, padamt), fftpad_psf(psf, padamt), adjoint=False))
            output_img = fftunpad(output_img, padamt)
            

        # Apply nonlinear activation
        if activation is not None:
            output_img = activation(output_img)

        return output_img
    
def split2d_layer(h_conv, rows, cols):
    # dims 1,2 of h_conv must be even multiple of rows,cols
    split_1d = tf.split(h_conv, num_or_size_splits=rows, axis=1)
    h_conv_split = tf.concat([tf.split(split_1d[i], num_or_size_splits=cols, axis=2) for i in range(rows)], 0)
    h_conv_split = tf.transpose(tf.squeeze(h_conv_split), [1, 2, 3, 0])
    return h_conv_split

def vis_h(h_conv, depth, rows, cols, name, buff=4):
        # this was for viewing multichannel convolution
        h_conv_split = tf.split(h_conv, depth, axis=3)
        h_conv_split = [tf.pad(h, [[0,0], [buff, buff], [buff, buff], [0,0]]) 
                       for h in h_conv_split]
        h_conv_tiled = tf.concat([tf.concat(h_conv_split[i*cols:(i+1)*cols], axis=2) for i in range(rows)], axis=1)
        tf.summary.image(name, h_conv_tiled, 3)
        
##############################
# Activation functions
##############################

def shifted_relu(v, thresh=10.):
    v = tf.cast(v, dtype=tf.float32)
    # shift = tf.reduce_mean(v)
    shift = tf.Variable(thresh, [1]) 
    tf.summary.scalar('relu_shift', tf.abs(shift))
    return tf.maximum(v, tf.abs(shift))

def shifted_relu_const(v, thresh=10.):
    v = tf.cast(v, dtype=tf.float32)
    tf.summary.scalar('relu_shift', tf.abs(thresh))
    return tf.maximum(v, tf.abs(shift))

def shifted_relu_zero(v, thresh=10.):
    v = tf.cast(v, dtype=tf.float32)
    #shift = tf.reduce_mean(v)
    shift = tf.Variable(thresh, [1]) 
    tf.summary.scalar('relu_shift', tf.abs(shift))
    return tf.maximum(0.0, v - tf.abs(shift))

def linear_trans(v):
    v = tf.cast(v, dtype=tf.float32)
    with tf.variable_scope('linear_trans'):
        a = tf.Variable(.1, [1])
        tf.summary.scalar('linear-a', tf.abs(a))
        b = tf.Variable(.1, [1])
        tf.summary.scalar('linear-b', b)

        v = tf.cast(v, dtype=tf.float32)
        transmittance = tf.add(tf.multiply(tf.abs(a), tf.abs(v)), tf.abs(b))
        transmittance = tf.clip_by_value(transmittance, 1e-3, 1)
        variable_summaries('transmittance', transmittance)
        return tf.multiply(v,transmittance)

def log_trans(v):
    v = tf.cast(v, dtype=tf.float32)
    with tf.variable_scope('log_trans'):
        a = tf.Variable(.05, [1])
        #a = .05
        tf.summary.scalar('log-a', tf.abs(a))
        b = tf.Variable(.8, [1])
        # b = .8
        tf.summary.scalar('log-b', b)

        transmittance = tf.add(tf.multiply(tf.abs(a), tf.log(tf.abs(v)+.2)), b)
        transmittance = tf.clip_by_value(transmittance, 1e-3, 1)
        variable_summaries('transmittance', transmittance)
        return tf.multiply(v,transmittance)
    
def sigmoid_trans(v):
    v = tf.cast(v, dtype=tf.float32)
    with tf.variable_scope('sigmoid_trans'):
        a = tf.Variable(.05, [1])
        # a = 1.0
        tf.summary.scalar('sig-a', tf.abs(a))
        b = tf.Variable(.8, [1])
        # b = 0.0
        tf.summary.scalar('sig-b', tf.abs(b))
        transmittance = 1/(1 + tf.exp(-tf.multiply(a, tf.add(tf.abs(v), b))))
        transmittance = tf.clip_by_value(transmittance, 1e-3, 1)
        variable_summaries('transmittance', transmittance)
        return tf.multiply(v,transmittance)
    




