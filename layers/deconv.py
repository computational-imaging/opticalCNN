import tensorflow as tf
import layers.optics as optics

def inverse_filter(blurred, estimate, psf, gamma):
     """Implements Weiner deconvolution in the frequency domain, with circular boundary conditions.


     Args:
         blurred: image with shape (batch_size, height, width, num_img_channels)
         estimate: image with shape (batch_size, height, width, num_img_channels)
         psf: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)

     TODO precompute OTF, adj_filt_img.
     """
     img_shape = blurred.shape.as_list()

     a_tensor_transp = tf.transpose(blurred, [0,3,1,2])
     estimate_transp = tf.transpose(estimate, [0,3,1,2])
     # Everything has shape (batch_size, num_channels, height, width)
     img_fft = tf.fft2d(tf.complex(a_tensor_transp, 0.))
     otf = optics.psf2otf(psf, output_size=img_shape[1:3])
     otf = tf.transpose(otf, [2,3,0,1])

     adj_conv = img_fft * tf.conj(otf)
     numerator = adj_conv + tf.fft2d(tf.complex(gamma*estimate_transp, 0.))

     kernel_mags = tf.square(tf.abs(otf))

     denominator = tf.complex(kernel_mags + gamma, 0.0)
     filtered = tf.div(numerator, denominator)
     cplx_result = tf.ifft2d(filtered)
     real_result = tf.real(cplx_result)
     # Get back to (batch_size, num_channels, height, width)
     result = tf.transpose(real_result, [0,2,3,1])
     return result

def resnet(img, num_layers, extra_ch=None, dilated=False, is_training=True):
    """ Implements residual CNN prior.

    Args:
        img: image with shape (batch_size, height, width, num_img_channels)
        num_layers: int, how many convolutional layers
        is_training: bool, in training mode?
    """
    ch = img.shape[3]
    if extra_ch is None:
      curr_x = img
    else:
      curr_x = tf.concat([img, extra_ch], axis=3)
    with tf.variable_scope('resnet'):
        for i in range(num_layers-1):
            # TODO batch norm.
            conv = tf.layers.conv2d(
                 inputs=curr_x,
                 filters=32,
                 kernel_size=[3, 3],
                 padding="same",
                 activation=tf.nn.relu)
            curr_x = conv

        conv = tf.layers.conv2d(
           inputs=curr_x,
           filters=ch,
           kernel_size=[3, 3],
           padding="same",
           activation=None)
        return conv + img
