import abc

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import ifftshift
import fractions

##############################
# Helper functions
##############################

def zoom(image_batch, zoom_fraction):
    """Get central crop of batch
    """
    images = tf.unstack(image_batch, axis=0)
    crops = []
    for image in images:
       crop = tf.image.central_crop(image, zoom_fraction)
       crops.append(crop)
    return tf.stack(crops, axis=0)

def transp_fft2d(a_tensor, dtype=tf.complex128):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    # Tensorflow's fft only supports complex64 dtype
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    a_tensor_transp = tf.transpose(a_tensor, [0,3,1,2])
    a_fft2d = tf.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a_fft2d, [0,2,3,1])
    return a_fft2d

def transp_ifft2d(a_tensor, dtype=tf.complex128):
    a_tensor = tf.transpose(a_tensor, [0,3,1,2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.ifft2d(a_tensor)
    # Transpose back to [batch_size, x, y, channels]
    a_ifft2d = tf.transpose(a_ifft2d_transp, [0,2,3,1])
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d

def compl_exp_tf(phase, dtype=tf.complex128, name='complex_exp'):
    """Complex exponent via euler's formula, since Cuda doesn't have a GPU kernel for that.
    Casts to *dtype*.
    """
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype),
                  1.j * tf.cast(tf.sin(phase), dtype=dtype),
                  name=name)

def laplacian_filter_tf(img_batch):
    """Laplacian filter. Also considers diagonals.
    """
    laplacian_filter = tf.constant([[1, 1, 1],[1,-8,1],[1,1,1]], dtype=tf.float32)
    laplacian_filter = tf.reshape(laplacian_filter, [3,3,1,1])

    filter_input = tf.cast(img_batch, tf.float32)
    filtered_batch = tf.nn.convolution(filter_input, filter=laplacian_filter, padding="SAME")
    return filtered_batch

def laplace_l1_regularizer(scale):
    if np.allclose(scale,0.):
       print("Scale of zero disables the laplace_l1_regularizer.")

    def laplace_l1(a_tensor):
        with tf.name_scope('laplace_l1_regularizer'):
            laplace_filtered = laplacian_filter_tf(a_tensor)
            laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
            attach_summaries("Laplace_filtered", tf.abs(laplace_filtered), image=True)
            return scale * tf.reduce_mean(tf.abs(laplace_filtered))

    return laplace_l1

def phaseshifts_from_height_map(height_map, wave_lengths, refractive_index):
    '''Calculates the phase shifts created by a height map with certain
    refractive index for light with specific wave length.
    '''
    # refractive index difference
    delta_N = refractive_index - 1.000277
    # wave number
    wave_nos = 2. * np.pi / wave_lengths
    # phase delay indiced by height field
    phi = wave_nos * delta_N * height_map
    phase_shifts = compl_exp_tf(phi)
    return phase_shifts

def get_one_phase_shift_thickness(wave_lengths, refractive_index):
    """Calculate the thickness (in meter) of a phaseshift of 2pi.
    """
    # refractive index difference
    delta_N = refractive_index - 1.
    # wave number
    wave_nos = 2. * np.pi / wave_lengths

    two_pi_thickness = (2. * np.pi) / (wave_nos * delta_N) * 1.1
    return two_pi_thickness

def attach_summaries(name, var, image=False, log_image=False):
    if image:
        tf.summary.image(name, var, max_outputs=3)
    if log_image:
        tf.summary.image(name+'_log', tf.log(var+1e-12), max_outputs=3)
    tf.summary.scalar(name+'_mean', tf.reduce_mean(var))
    tf.summary.scalar(name+'_max', tf.reduce_max(var))
    tf.summary.scalar(name+'_min', tf.reduce_min(var))
    # tf.summary.histogram(name+'_histogram', var)

    # Attaching a tensor summary will allow us to retrieve the actual value of the
    # height map just from the summary
    tf.summary.tensor_summary(name, var)

def attach_img(name, var):
    tf.summary.image(name, var, max_outputs=3)
    tf.summary.scalar(name+'_mean', tf.reduce_mean(var))
    tf.summary.scalar(name+'_max', tf.reduce_max(var))
    tf.summary.scalar(name+'_min', tf.reduce_min(var))


def fftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        split = (input_shape[axis] + 1)//2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def ifftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor

def psf2otf(input_filter, output_size):
    """Convert 4D tensorflow filter into its FFT.
    """
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    fh, fw, _, _ = input_filter.shape.as_list()
    # Circular shift.
    padded = tf.pad(input_filter, [[0, output_size[0]-fh],
                                  [0, output_size[1]-fw], [0,0], [0,0]], "CONSTANT")
    # Shift left.
    left = padded[:, 0:(fw-1)//2,:,:]
    right = padded[:, (fw-1)//2:,:,:]
    padded = tf.concat([right, left], 1)
    # Shift down.
    up = padded[0:(fh-1)//2, :,:,:]
    down = padded[(fh-1)//2:, :,:,:]
    padded = tf.concat([down, up], 0)
    # Take FFT
    tmp = tf.transpose(padded, [2,3,0,1])
    tmp = tf.fft2d(tf.complex(tmp, 0.))
    return tf.transpose(tmp, [2,3,0,1])


def fft_conv2d(img, psf, adjoint=False):
    """Implements convolution in the frequency domain, with circular boundary conditions.


    Args:
        img: image with shape (batch_size, height, width, num_img_channels)
        psf: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)
    """
    img = tf.cast(img, dtype=tf.float32)
    psf = tf.cast(psf, dtype=tf.float32)

    img_shape = img.shape.as_list()
    img = fftshift2d_tf(img)
    img_fft = transp_fft2d(img) # (batch_size, num_channels, height, width)

    otf = psf2otf(psf, output_size=img_shape[1:3])
    otf = tf.cast(otf, tf.complex128)
    otf = tf.transpose(otf, [2,0,1,3])

    if adjoint:
      result = transp_ifft2d(img_fft * otf)
    else:
      result = transp_ifft2d(img_fft * tf.conj(otf))

    result = ifftshift2d_tf(result)
    return tf.cast(tf.real(result), tf.float64)


def depth_dep_convolution(img, psfs, disc_depth_map, use_fft=True):
    """

    Args:
        img: image with shape (batch_size, height, width, num_img_channels)
        psfs: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)
        disc_depth_map: The discretized depth map.
        use_fft: Use fft_conv2d or normal conv2d
    """
    # TODO: only convolve with PSFS that are necessary.
    img = tf.cast(img, dtype=tf.float32)
    psfs = tf.cast(psfs, dtype=tf.float32)
    disc_depth_map = tf.cast(disc_depth_map, tf.int16)

    input_shape = img.shape.as_list()

    if use_fft:
        blurred_imgs = fft_conv2d(img, psfs)
    else:
        blurred_imgs = tf.nn.convolution(img, filter=psfs, padding="SAME")

    # Pick each pixel from the blur volume according to the disc_depth_map
    spatial_indices = np.mgrid[:input_shape[1], :input_shape[2]].transpose(1,2,0).astype(np.int16)
    spatial_indices = tf.tile(spatial_indices[None,:,:,:], [input_shape[0],1,1,1])

    batch_indices = np.arange(input_shape[0])[:,None, None, None]
    batch_indices = np.tile(batch_indices, [1, input_shape[1], input_shape[2], 1]).astype(np.int16)

    disc_depth_map_idcs = tf.concat([batch_indices, spatial_indices, disc_depth_map], axis=3)
    disc_depth_map_idcs = tf.cast(disc_depth_map_idcs, tf.int32)

    result = tf.expand_dims(tf.gather_nd(blurred_imgs, disc_depth_map_idcs, name='depth_dependent_convolution'), -1)

    return result

def get_spherical_wavefront_phase(resolution,
                                  physical_size,
                                  wave_lengths,
                                  source_distance):
    source_distance = tf.cast(source_distance, tf.float64)
    physical_size = tf.cast(physical_size, tf.float64)
    wave_lengths = tf.cast(wave_lengths, tf.float64)

    N, M = resolution
    [x, y] = np.mgrid[-N//2:N//2,
                      -M//2:M//2].astype(np.float64)

    x = x/N * physical_size
    y = y/M * physical_size

    # Assume distance to source is approx. constant over wave
    curvature = tf.sqrt(x**2 + y**2 + source_distance**2)
    wave_nos = 2. * np.pi / wave_lengths

    phase_shifts = compl_exp_tf(wave_nos * curvature)
    phase_shifts = tf.expand_dims(tf.expand_dims(phase_shifts, 0), -1)
    return phase_shifts


def least_common_multiple(a, b):
    return abs(a * b) / fractions.gcd(a,b) if a and b else 0

def area_downsampling_tf(input_image, target_side_length):
    input_shape = input_image.shape.as_list()
    input_image = tf.cast(input_image, tf.float32)

    if not input_shape[1] % target_side_length:
        factor = int(input_shape[1]/target_side_length)
        output_img = tf.nn.avg_pool(input_image,
                                    [1,factor,factor,1],
                                    strides=[1,factor,factor,1],
                                    padding="VALID")
    else:
        # We upsample the image and then average pool
        lcm_factor = least_common_multiple(target_side_length, input_shape[1]) / target_side_length

        if lcm_factor>10:
            print("Warning: area downsampling is very expensive and not precise if source and target wave length have a large least common multiple")
            upsample_factor=10
        else:
            upsample_factor = int(lcm_factor)

        img_upsampled = tf.image.resize_nearest_neighbor(input_image,
                                                         size=2*[upsample_factor*target_side_length])
        output_img = tf.nn.avg_pool(img_upsampled,
                                    [1,upsample_factor,upsample_factor,1],
                                    strides=[1,upsample_factor,upsample_factor,1],
                                    padding="VALID")

    return output_img


def get_intensities(input_field):
    return tf.square(tf.abs(input_field), name='intensities')

def gaussian_noise(image, stddev=0.001):
    dtype = image.dtype
    return image + tf.random_normal(image.shape, 0.0, stddev,dtype=dtype)

##################################
# Optical elements & Propagation
##################################

class Propagation(abc.ABC):
    def __init__(self,
                 input_shape,
                 distance,
                 discretization_size,
                 wave_lengths,
                 pad_width,
                 pad_mode):

        self.input_shape = input_shape
        self.distance = distance
        self.wave_lengths = wave_lengths
        self.pad_width = pad_width
        self.pad_mode = pad_mode
        self.wave_nos = 2. * np.pi / wave_lengths
        self.discretization_size = discretization_size

    @abc.abstractmethod
    def _propagate(self, input_field):
        """Propagate an input field through the medium
        """

    def __call__(self, input_field):
        return self._propagate(input_field)


class FresnelPropagation(Propagation):
    def _propagate(self, input_field):
        _, M_orig, N_orig, _ = self.input_shape
        # zero padding.
        Mpad = M_orig//4
        Npad = N_orig//4
        M = M_orig + 2*Mpad
        N = N_orig + 2*Npad
        padded_input_field = tf.pad(input_field,
          [[0,0], [Mpad,Mpad], [Npad,Npad],[0,0]])

        [x,y] = np.mgrid[-N//2:N//2,
                         -M//2:M//2]

        # Spatial frequency
        fx = x / (self.discretization_size*N) # max frequency = 1/(2*pixel_size)
        fy = y / (self.discretization_size*M)

        # We need to ifftshift fx and fy here, because ifftshift doesn't exist in TF.
        fx = ifftshift(fx)
        fy = ifftshift(fy)

        fx = fx[None,:,:,None]
        fy = fy[None,:,:,None]

        squared_sum = np.square(fx) + np.square(fy)

        # We create a non-trainable variable so that this computation can be reused
        # from call to call.
        tmp = tf.float64(self.wave_lengths*np.pi*-1.*squared_sum)
        constant_exp_part_init = tf.constant_initializer(tmp)
        constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part",
                                                 initializer=constant_exp_part_init,
                                                 shape=padded_input_field.shape,
                                                 dtype=tf.float64,
                                                 trainable=False)

        H = compl_exp_tf( self.distance * constant_exponent_part, dtype=tf.complex128,
                          name='fresnel_kernel')

        objFT = transp_fft2d( padded_input_field )
        out_field = transp_ifft2d( objFT * H)

        return out_field[:,Mpad:-Mpad,Npad:-Npad,:]

class AmplitudeMask():
    def __init__(self,
                 amp_mask):
        self.amp_mask = tf.convert_to_tensor(amp_mask, dtype=tf.complex128)

    def __call__(self, input_field):
        return input_field * self.amp_mask

def bacteriorhodopsin_nonlinearity(input_field, trainable=False):
    a = tf.get_variable('a',
                        shape=[],
                        initializer=tf.constant_initializer(0.8178),
                        trainable=trainable)

    b = tf.get_variable('b',
                        shape=[],
                        initializer=tf.constant_initializer(0.03047),
                        trainable=trainable)

    amplitude = tf.cast(tf.abs(input_field), tf.float32)

    phase = tf.atan2(tf.imag(input_field), tf.real(input_field))
    amplitude_nl = tf.cast(a + b*tf.log(amplitude), tf.complex128)

    return amplitude_nl * compl_exp_tf(phase)


class PhasePlate():
    def __init__(self,
                 wave_lengths,
                 height_map,
                 refractive_index,
                 phase_shifts=None,
                 height_tolerance=None,
                 lateral_tolerance=None):

        self.wave_lengths = wave_lengths
        self.height_map = height_map
        self.refractive_index=refractive_index
        self.height_tolerance=height_tolerance
        self.lateral_tolerance=lateral_tolerance
        if phase_shifts is not None:
            self.phase_shifts=compl_exp_tf(phase_shifts)

        self._build()

    def _build(self):
        # Add manufacturing tolerances in the form of height map noise
        if self.height_tolerance is not None:
            self.height_map += tf.random_uniform(shape=self.height_map.shape,
                                                 minval=-self.height_tolerance,
                                                 maxval=self.height_tolerance,
                                                 dtype=tf.float64)
            print("Phase plate with manufacturing tolerance %0.2e"%self.height_tolerance)

        # TODO: Add manufacturing tolerance on lateral shift

        if self.height_map is not None:
            self.phase_shifts = phaseshifts_from_height_map(self.height_map,
                                                        self.wave_lengths,
                                                        self.refractive_index)

    def __call__(self, input_field):
        input_field = tf.cast(input_field, tf.complex128)
        return tf.multiply(input_field, self.phase_shifts, name='phase_plate_shift')


def propagate_fresnel(input_field,
                      distance,
                      input_sample_interval,
                      wave_lengths,
                      pad_width=None,
                      pad_mode=None):
    # TODO: Implement padding
    input_shape = input_field.shape.as_list()
    propagation = FresnelPropagation(input_shape,
                                     distance=distance,
                                     discretization_size=input_sample_interval,
                                     wave_lengths=wave_lengths,
                                     pad_width=pad_width,
                                     pad_mode=pad_mode)
    return propagation(input_field)

def square_aperture(input_field, aperture_fraction):
    input_shape = input_field.shape.as_list()
    [x, y] = np.mgrid[-input_shape[1] // 2: input_shape[1] // 2,
                      -input_shape[2] // 2: input_shape[2] // 2].astype(np.float64)

    max_val = np.amax(x)

    r = np.maximum(np.abs(x), np.abs(y))
    aperture = (r < aperture_fraction * max_val)[None,:,:,None]
    plt.imshow(aperture.squeeze())
    plt.show()
    element = AmplitudeMask(amp_mask=aperture)
    return element(input_field)


def circular_aperture(input_field, max_val=None):
    input_shape = input_field.shape.as_list()
    [b, x, y, c] = np.mgrid[0:input_shape[0],
                      -input_shape[1] // 2: input_shape[1] // 2,
                      -input_shape[2] // 2: input_shape[2] // 2,
                      0:input_shape[3]].astype(np.float64)

    if max_val is None:
        max_val = np.amax(x)

    r = np.sqrt(x ** 2 + y ** 2) # [None,:,:,None]
    return tf.where(r<max_val,
                    input_field,
                    tf.zeros(shape=input_shape,dtype=input_field.dtype))


def phase_shift_element_old(input_field,
                        name,
                        block_size=1,
                        phase_shift_initializer=None,
                        phase_shift_regularizer=None,
                        refractive_index=1.5):
        map_shape = input_field.shape.as_list()
        b, h, w, c = map_shape
        input_shape = [b, h//block_size, w//block_size, c]

        if phase_shift_initializer is None:
            init_phase_shift_value = np.ones(shape=input_shape, dtype=np.float64)
            phase_shift_initializer = tf.constant_initializer(init_phase_shift_value)

        with tf.variable_scope(name, reuse=False):
            phase_shift_var = tf.get_variable(name="phase_shifts",
                                             shape=input_shape,
                                             dtype=tf.float64,
                                             trainable=True,
                                             initializer=phase_shift_initializer)

            phase_shifts_full = tf.image.resize_images(phase_shift_var, map_shape[1:3],
                                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            if phase_shift_regularizer is not None:
                tf.contrib.layers.apply_regularization(phase_shift_regularizer, weights_list=[phase_shifts_full])
            attach_summaries("Phase_map", phase_shifts_full, image=True)

            phase_shifts_exp= compl_exp_tf(phase_shifts_full)

        input_field = tf.cast(input_field, tf.complex128)
        return tf.multiply(input_field, phase_shifts_exp, name='phase_plate_shift')

def phase_shift_element(map_shape,
                       name,
                       wave_lengths,
                       block_size=1,
                       phase_shift_initializer=None,
                       phase_shift_regularizer=None,
                       height_tolerance=None, # Default height tolerance is 2 nm.
                       refractive_index=1.5):

        b, h, w, c = map_shape
        input_shape = [b, h//block_size, w//block_size, c]

        if phase_shift_initializer is None:
            init_phase_shift_value = np.ones(shape=input_shape, dtype=np.float64) * 1e-4
            phase_shift_initializer = tf.constant_initializer(init_phase_shift_value)

        with tf.variable_scope(name, reuse=False):
            phase_shift_var = tf.get_variable(name="phase_shift_sqrt",
                                             shape=input_shape,
                                             dtype=tf.float64,
                                             trainable=True,
                                             initializer=phase_shift_initializer)

            phase_shift_full = tf.image.resize_images(phase_shift_var, map_shape[1:3],
                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            phase_shifts = tf.square(phase_shift_full, name='phase_shift')

            if phase_shift_regularizer is not None:
                tf.contrib.layers.apply_regularization(phase_shift_regularizer, weights_list=[phase_shifts])

            attach_summaries("phase_shift", phase_shifts, image=True)

        element =  PhasePlate(wave_lengths=wave_lengths,
                              height_map=None,
                              phase_shifts=phase_shifts,
                              refractive_index=refractive_index,
                              height_tolerance=height_tolerance)

        return element


def height_map_element(map_shape,
                       name,
                       wave_lengths,
                       block_size=1,
                       height_map_initializer=None,
                       height_map_regularizer=None,
                       height_tolerance=None, # Default height tolerance is 2 nm.
                       refractive_index=1.5):

        b, h, w, c = map_shape
        input_shape = [b, h//block_size, w//block_size, c]

        if height_map_initializer is None:
            init_height_map_value = np.ones(shape=input_shape, dtype=np.float64) * 1e-4
            height_map_initializer = tf.constant_initializer(init_height_map_value)

        with tf.variable_scope(name, reuse=False):
            height_map_var = tf.get_variable(name="height_map_sqrt",
                                             shape=input_shape,
                                             dtype=tf.float64,
                                             trainable=True,
                                             initializer=height_map_initializer)

            height_map_full = tf.image.resize_images(height_map_var, map_shape[1:3],
                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            height_map = tf.square(height_map_full, name='height_map')

            if height_map_regularizer is not None:
                tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

            attach_summaries("Height_map", height_map, image=True)

        element =  PhasePlate(wave_lengths=wave_lengths,
                              height_map=height_map,
                              refractive_index=refractive_index,
                              height_tolerance=height_tolerance)

        return element

#------ Fourier coefficient phase mask  ------#
def fourier_element(map_shape,
                    name,
                    wave_lengths,
                    refractive_index,
                    frequency_range = 0.5,
                    height_map_regularizer=None,
                    height_tolerance=None, # Default height tolerance is 2 nm.
                   ):
    _, height, width, _ = map_shape
    height_map_shape = [1, height, width, 1]

    fourier_initializer=tf.zeros_initializer()

    with tf.variable_scope(name, reuse=False):
        fourier_vars_real = tf.get_variable('fourier_coeffs_real',
                                       shape=[1, int(height*frequency_range), int(width*frequency_range), 1],
                                       dtype=tf.float32,
                                       trainable=True,
                                       initializer=fourier_initializer)
        fourier_vars_cplx = tf.get_variable('fourier_coeffs_cmplx',
                                       shape=[1, int(height*frequency_range), int(width*frequency_range), 1],
                                       dtype=tf.float32,
                                       trainable=True,
                                       initializer=fourier_initializer)
        fourier_coeffs = tf.complex(fourier_vars_real, fourier_vars_cplx)
        attach_summaries("Fourier_coeffs", tf.abs(fourier_coeffs), image=True, log_image=False)
        size = int(height*frequency_range)
        padding_width_1 = int(np.ceil((height - size)/2))
        padding_width_2 = int(np.floor((height - size)/2))
        fourier_coeffs_padded = tf.pad(fourier_coeffs, [[0,0],[padding_width_1,padding_width_2],[padding_width_1,padding_width_2],[0,0]])
        # print(fourier_coeffs_padded.shape.as_list())
        height_map = tf.real(transp_ifft2d(ifftshift2d_tf(fourier_coeffs_padded)))

        if height_map_regularizer is not None:
            tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        attach_summaries("Height_map", height_map, image=True, log_image=True)

    element =  PhasePlate(wave_lengths=wave_lengths,
                          height_map=height_map,
                          refractive_index=refractive_index,
                          height_tolerance=height_tolerance)

    return element


#------ Zernike basis phase mask object ------#
def zernike_element(zernike_volume,
                    name,
                    wavelengths,
                    refractive_idcs,
                    r_NA,
                    zernike_initializer=None,
                    height_map_regularizer=None,
                    height_tolerance=None, # Default height tolerance is 2 nm.
                    zernike_scale=1e5):

    _, height, width = zernike_volume.shape.as_list()
    height_map_shape = [1, height, width, 1]
    num_zernike_coeffs = zernike_volume.shape.as_list()[0]

    if zernike_initializer is None:
        zernike_initializer = tf.zeros_initializer()
        # zernike_initializer = tf.random_normal_initializer(stddev=1e-6)

    with tf.variable_scope(name, reuse=False):
        zernike_coeffs = tf.get_variable('zernike_coeffs',
                                       shape=[num_zernike_coeffs, 1, 1],
                                       dtype=tf.float32,
                                       trainable=True,
                                       initializer=zernike_initializer)
        #mask = np.ones([num_zernike_coeffs, 1, 1])
        #mask[0] = 0.
        #zernike_coeffs *= mask/zernike_scale

        for i in range(num_zernike_coeffs):
            tf.summary.scalar('zernike_coeff_%d'%i, tf.squeeze(zernike_coeffs[i,:,:]))

        height_map = tf.reduce_sum(zernike_coeffs*zernike_volume, axis=0)
        height_map = tf.expand_dims(tf.expand_dims(height_map, 0), -1, name='height_map')
        height_map = circular_aperture(height_map, max_val = r_NA)

        if height_map_regularizer is not None:
            tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        height_map_summary = (height_map - tf.reduce_min(height_map))/(tf.reduce_max(height_map) - tf.reduce_min(height_map))
        attach_summaries("z_height_map", height_map_summary, image=True)

    element =  PhasePlate(wavelengths,
                          height_map=height_map,
                          refractive_index=refractive_idcs,
                          height_tolerance=height_tolerance)

    return element


#------ AmplitudeMask object ------#
class AmplitudeMask():
    def __init__(self, amplitude_map):

        self.amplitude_map = amplitude_map

        self._build()

    def _build(self):
        self.amplitude_map = tf.cast(self.amplitude_map, tf.complex128)

    def __call__(self, input_field):
        input_field = tf.cast(input_field, tf.complex128)
        return tf.multiply(input_field, self.amplitude_map, name='apply_amplitude_mask')

def amplitude_map_element(map_shape, r_NA,
                       name,
                       block_size=1,
                       amplitude_map_initializer=None,
                       amplitude_map_regularizer=None):

        b, h, w, c = map_shape
        input_shape = [b, h//block_size, w//block_size, c]

        if amplitude_map_initializer is None:
            init_amplitude_map_value = np.ones(shape=input_shape, dtype=np.float64)
            amplitude_map_initializer = tf.constant_initializer(init_amplitude_map_value)

        with tf.variable_scope(name, reuse=False):
            amplitude_map_var = tf.get_variable(name="amplitude_map_sqrt",
                                             shape=input_shape,
                                             dtype=tf.float64,
                                             trainable=True,
                                             initializer=amplitude_map_initializer)

            amplitude_map_full = tf.image.resize_images(amplitude_map_var, map_shape[1:3],
                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            amplitude_map_full = circular_aperture(amplitude_map_full, max_val = r_NA)
            amplitude_map = tf.square(amplitude_map_full, name='amplitude_map')


            if amplitude_map_regularizer is not None:
                tf.contrib.layers.apply_regularization(amplitude_map_regularizer, weights_list=[amplitude_map])

            attach_summaries("amplitude_map", amplitude_map, image=True)

        element =  AmplitudeMask(amplitude_map=amplitude_map)

        return element

def plano_convex_lens(input_field,
                      focal_length,
                      wave_lengths,
                      discretization_step,
                      refractive_index=1.5):

    input_shape = input_field.shape.as_list()
    _, N, M, _ = input_shape

    convex_radius = (refractive_index - 1.) * focal_length
    [x, y] = np.mgrid[-N//2:N//2,
                      -M//2:M//2].astype(np.float64)

    x = x * discretization_step
    y = y * discretization_step

    # This approximates the spherical surface with qaudratic-phase surfaces.
    height_map = - (x ** 2 + y ** 2) / 2. * (1. / convex_radius)
    height_map = height_map[None,:,:,None]

    element = PhasePlate(wave_lengths=wave_lengths,
                         height_map=height_map,
                         refractive_index=refractive_index)

    return element(input_field)


def plano_convex_initializer(focal_length,
                             wave_lengths,
                             discretization_step,
                             refractive_index=1.5):
    def _initializer(shape, dtype=tf.float64, **kwargs):
        convex_radius = (refractive_index - 1.) * focal_length
        _,N,M,_ = shape
        [x, y] = np.mgrid[-N//2:N//2,
                          -M//2:M//2].astype(np.float64)

        x = x * discretization_step
        y = y * discretization_step

        # This approximates the spherical surface with qaudratic-phase surfaces.
        height_map = - (x ** 2 + y ** 2) / 2. * (1. / convex_radius)
        height_map = height_map[None,:,:,None]
        return tf.constant(height_map, dtype=dtype)
    return _initializer


def cubic_phase_shifts(input_field):
    input_shape = input_field.shape.as_list()
    _, N, M, _ = input_shape

    [x, y] = np.mgrid[-N//2:N//2,
                      -M//2:M//2].astype(np.float64)

    x = x / np.array(N//2, dtype=np.float64)
    y = y / np.array(M//2, dtype=np.float64)

    phase_shifts = np.mod( 20. * np.pi * ( x**3 + y**3 ) , 2. * np.pi )
    phase_shifts = phase_shifts[None, :, :, None]

    phase_shifts = compl_exp_tf(phase_shifts, dtype=tf.complex128)
    input_field = tf.cast(input_field, tf.complex128)

    return input_field * phase_shifts


def propagate_2f_collimator(input_field):
    return transp_fft2d(input_field)

class Sensor():
    def __init__(self, resolution, input_is_intensities=False):
        self.resolution = resolution
        self.input_is_intensities = input_is_intensities

    def __call__(self, input_field):
        if not self.input_is_intensities:
            sensor_readings = tf.square(tf.abs(input_field))
        else:
            sensor_readings = input_field
        sensor_readings = tf.cast(sensor_readings, tf.float64,
                                  name='sensor_readings')

        sensor_readings = area_downsampling_tf(sensor_readings, self.resolution[0])

        return sensor_readings

def gaussian_noise(image, stddev=0.001):
    dtype = image.dtype
    return image + tf.random_normal(image.shape, 0.0, stddev,dtype=dtype)

class OpticalSystem(abc.ABC):
    def __init__(self,
                 forward_model,
                 wave_resolution,
                 wave_lengths,
                 sensor_resolution,
                 noise_model=gaussian_noise,
                 psf_resolution=None,
                 target_distance=None,
                 discretization_size=1e-6,
                 use_planar_incidence=True,
                 upsample=True,
                 depth_bins=None):

        self.forward_model = forward_model
        self.wave_resolution = wave_resolution
        self.wave_lengths = wave_lengths
        self.use_planar_incidence=use_planar_incidence
        self.depth_bins = depth_bins
        self.discretization_size = discretization_size
        self.sensor_resolution = sensor_resolution
        self.noise_model = noise_model
        self.upsample=upsample
        self.target_distance=target_distance

        if psf_resolution is None:
           psf_resolution = wave_resolution
        self.psf_resolution = psf_resolution

        self.get_psfs()
        self.pixel_size = discretization_size * np.array(wave_resolution)/np.array(sensor_resolution)

    def get_psfs(self):
        # Sort the point source distances in increasing order
        if self.use_planar_incidence:
            input_fields = [tf.ones(self.wave_resolution, dtype=tf.float64)[None,:,:,None]]
        else:
            distances = self.depth_bins

            if self.target_distance is not None:
                distances += [self.target_distance]

            physical_size = float(self.wave_resolution[0] * self.discretization_size)

            N, M = self.wave_resolution
            [x, y] = np.mgrid[-N//2:N//2,
                              -M//2:M//2].astype(np.float64)

            x = x/N * physical_size
            y = y/M * physical_size

            squared_sum = x**2 + y**2

            wave_nos = 2. * np.pi / self.wave_lengths

            input_fields = []
            for distance in distances:
                # Assume distance to source is approx. constant over wave
                curvature = tf.sqrt(squared_sum + tf.cast(distance, tf.float64)**2)

                spherical_wavefront = compl_exp_tf(wave_nos * curvature)
                spherical_wavefront = tf.expand_dims(tf.expand_dims(spherical_wavefront, 0), -1)

                input_fields.append(spherical_wavefront)


        psfs = []
        with tf.variable_scope("Forward_model") as scope:
            for input_field in input_fields:
                sensor_incident_field = self.forward_model(input_field)
                psf = get_intensities(sensor_incident_field)
                if not self.upsample:
                    psf = area_downsampling_tf(psf, self.sensor_resolution[0])
                psf /= tf.reduce_sum(psf)
                psf = tf.cast(psf, tf.float64)
                psfs.append(tf.transpose(psf, [1,2,0,3])) # (Height, width, 1, 1)
                scope.reuse_variables()

        if self.target_distance is not None:
            self.target_psf = psfs.pop()
            attach_summaries('target_psf', tf.transpose(self.target_psf, [3,0,1,2]), image=True)

        # Concatenate the psfs such that they can be used as a tensorflow filter (h, w, in_channels, out_channels)
        psfs = tf.concat(psfs, axis=3) # (Height, width, 1, self.depth_bins)

        attach_summaries('PSFS', tf.transpose(psfs, [3,0,1,2]), image=True)
        self.psfs = psfs


    def get_sensor_img(self,
                       input_img,
                       noise_sigma=0.001,
                       depth_dependent=False,
                       depth_map=None):
        """"""
        # Upsample input_img to match wave resolution.
        if self.upsample:
            input_img = tf.image.resize_images(input_img, self.wave_resolution,
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if depth_dependent:
            if self.upsample:
                depth_map = tf.image.resize_images(depth_map, self.wave_resolution,
                                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            sensor_img = depth_dep_convolution(input_img, self.psfs, disc_depth_map=depth_map)
        else:
            sensor_img = fft_conv2d(input_img, self.psfs)

        # Down sample measured image to match sensor resolution.
        if self.upsample:
            sensor_img = area_downsampling_tf(sensor_img, self.sensor_resolution[0])
        noisy_img = self.noise_model(sensor_img, noise_sigma)
        print("Additive noise of %0.2e"%noise_sigma)
        attach_summaries("Sensor_img", noisy_img, image=True, log_image=False)
        return noisy_img

