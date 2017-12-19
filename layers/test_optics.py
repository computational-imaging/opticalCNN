import layers.optics as optics
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import cv2
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def cv2_load_rgb(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    img /= 255.
    return img

def test_convex_height_map(propagation_type='two_step_fresnel'):
    '''Tests if the psf of a pano-convex lens is computed correctly.
    '''
    tf.reset_default_graph()

    with tf.device('/device:GPU:2'):
        sensor_resolution = (512, 512)
        wave_resolution = (2400, 2400)
        focal_length = 5e-2
        wave_length = 532e-9
        discretization_step = 1e-6

        # Get the phase shifts (not exponential, linear phase shifts!)
        distance_graph = tf.placeholder(tf.float32, shape=[])

        input_field = tf.ones([1,wave_resolution[0],wave_resolution[1],1])
        field = optics.plano_convex_lens(input_field,
                                         focal_length=focal_length,
                                         wave_lengths=wave_length,
                                         discretization_step=discretization_step)
        field = optics.circular_aperture(field)
        field = optics.propagate_fresnel(field,
                                         distance=distance_graph,
                                         input_sample_interval=discretization_step,
                                         wave_lengths=wave_length)
        psf_graph = optics.Sensor(resolution=(2400,2400), input_is_intensities=False)(field)

        distances = np.linspace(1e-6, 5 * focal_length, 50)
        psf_slice = np.zeros([wave_resolution[0], len(distances)])
        time_start = time.time()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            for idx, distance in enumerate(distances):
                feed_dict = {distance_graph: distance}

                psf = sess.run(psf_graph, feed_dict=feed_dict)
                psf = psf.squeeze().astype(np.float64)

                psf_slice[:, idx] = psf[:, wave_resolution[1] // 2]

                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(np.log(psf + 1e-9))
                ax[1].imshow(np.log(psf_slice + 1e-9))
                ax[0].set_title('Distance %0.2e' % distance)
                plt.show()
                plt.close(fig)
        time_end = time.time()
        print(time_end - time_start)

def test_block_vars():
    # Test that height map is properly blocked out.
    tf.reset_default_graph()

    wave_resolution = np.array((512,512))
    sensor_resolution = (512,512)
    wave_length = 532e-9
    discretization_step = 1e-6
    distance = 1e-2
    refractive_index=1.5
    np.random.seed(1)
    with tf.device('/device:GPU:2'):
        input_field = tf.ones([1,wave_resolution[0],wave_resolution[1],1])
        init_map = np.random.uniform(size=[1,wave_resolution[0]//4,wave_resolution[1]//4,1])
        init_map = tf.constant_initializer(init_map.astype(np.float64))
        field = optics.height_map_element(input_field,
                                          wave_lengths=wave_length,
                                          height_map_regularizer=optics.laplace_l1_regularizer(1),
                                          block_size=4,
                                          height_map_initializer=init_map,
                                          name='height_map_optics')

    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        eval_field = sess.run(field)
        sample1 = eval_field[0,::4,::4,0]
        sample2 = eval_field[0,1::4,1::4,0]
        sample3 = eval_field[0,2::4,3::4,0]
        assert np.allclose(sample1, sample2)
        assert np.allclose(sample3, sample2)

def test_train_collimator():
    # Test if we can supervise the PSF with the known psf of a convex lens to
    # learn a collimator lens.
    tf.reset_default_graph()

    wave_resolution = np.array((512,512))
    sensor_resolution = (512,512)
    wave_length = 532e-9
    discretization_step = 1e-6
    distance = 1e-2
    refractive_index=1.5

    # Define the model and place it on the GPU
    with tf.device('/device:GPU:2'):
        input_field = tf.ones([1,wave_resolution[0],wave_resolution[1],1])
        # Get the Ground-Truth PSF of a convex lens
        field = optics.plano_convex_lens(input_field,
                                         focal_length=distance,
                                         wave_lengths=wave_length,
                                         discretization_step=discretization_step)
        field = optics.circular_aperture(field)
        field = optics.propagate_fresnel(field,
                                         distance=distance,
                                         input_sample_interval=discretization_step,
                                         wave_lengths=wave_length)
        gt_psf_graph = optics.Sensor(resolution=sensor_resolution,
                                     input_is_intensities=False)(field)

        # Get the learned PSF
        field = optics.height_map_element(input_field,
                                          wave_lengths=wave_length,
                                          height_map_regularizer=optics.laplace_l1_regularizer(1),
                                          name='height_map_optics')
        field = optics.circular_aperture(field)
        field = optics.propagate_fresnel(field,
                                         distance=distance,
                                         input_sample_interval=discretization_step,
                                         wave_lengths=wave_length)
        psf_graph = optics.Sensor(resolution=sensor_resolution,
                                   input_is_intensities=False)(field)

        data_loss = (tf.losses.mean_squared_error(tf.squeeze(psf_graph), tf.squeeze(gt_psf_graph)))
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = tf.add(data_loss, reg_loss)

    # Configure the optimizer and the learning rate
    starter_learning_rate = 1e-5
    optimizer = tf.train.AdamOptimizer(learning_rate=starter_learning_rate,
                                       beta1=0.8,
                                       beta2=0.99,
                                       epsilon=1.)
    train_step = optimizer.minimize(total_loss)

    # Train
    init = tf.global_variables_initializer()

    height_map_graph = tf.get_default_graph().get_tensor_by_name(name='height_map_optics/height_map:0')

    loss_values = []
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        for step in range(10000):
            _, total_loss_val, data_loss_val, reg_loss_val = sess.run([train_step, total_loss, data_loss, reg_loss])
            loss_values.append(data_loss_val + reg_loss_val)

            if not step%500:
                hm, psf, gt_psf = sess.run([height_map_graph, psf_graph, gt_psf_graph])

                hm = hm.squeeze()

                print(step, total_loss_val, data_loss_val, reg_loss_val)
                fig,ax = plt.subplots(1,3)
                ax[0].imshow(hm.squeeze())
                ax[0].axis('off')
                ax[1].imshow(np.log(psf.squeeze() + 1e-9))
                ax[1].axis('off')
                ax[2].imshow(np.log(np.abs(psf.squeeze() - gt_psf.squeeze())+1e-9))
                ax[2].axis('off')
                plt.show()

                plt.plot(hm[hm.shape[0]//2, :])
                plt.show()
                plt.close(fig)

def test_spherical_wavefront_phase():
    tf.reset_default_graph()

    wave_resolution = np.array((512,512))
    sensor_resolution = (512,512)
    wave_lengths = 532e-9
    input_sample_interval = 1e-6
    refractive_index=1.5


    distances = np.linspace(0.01, 2., 25)

    distance_graph = tf.placeholder(tf.float32, shape=[])

    spherical_wavefront_graph = optics.get_spherical_wavefront_phase(wave_resolution,
                                                                     wave_resolution[0] * input_sample_interval,
                                                                     wave_lengths=wave_lengths,
                                                                     source_distance=distance_graph)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for distance in distances:
            wavefront_phase = sess.run(spherical_wavefront_graph, feed_dict={distance_graph:distance})
            wavefront_angle = np.angle(wavefront_phase.squeeze())

            print(distance)
            print(np.amax(wavefront_angle))
            print(np.amin(wavefront_angle))
            print(np.std(wavefront_angle))
            plt.imshow(wavefront_angle)
            plt.show()


def test_optical_element():
    tf.reset_default_graph()

    wave_resolution = np.array((512,512))
    sensor_resolution = (512,512)
    wave_lengths = 532e-9
    input_sample_interval = 1e-6
    distance = 1e-2
    refractive_index=1.5

    depth_bins = sorted(np.linspace(distance*2, distance*10, 32))
    discretization_step=depth_bins

    input_img = np.zeros((512,512))
    input_img[:,255] = 1.
    input_img = input_img[None,:,:,None].astype(np.float32)
    plt.imshow(input_img.squeeze())
    plt.show()

    test_depth = np.concatenate([np.ones((512//len(depth_bins), 512)) * depth for depth in depth_bins], axis=0)[None,:,:,None]
    test_depth = np.digitize(test_depth, depth_bins,right=True).astype(np.int32)
    plt.imshow(test_depth.squeeze())
    plt.show()

    with tf.device('/device:GPU:2'):
        # Convert to tensors
        test_depth = tf.convert_to_tensor(test_depth, tf.int32)
        input_img = tf.convert_to_tensor(input_img, tf.float32)

        def forward_model(input_field):
            field = optics.plano_convex_lens(input_field,
                                             focal_length=distance,
                                             wave_lengths=wave_lengths,
                                             discretization_step=input_sample_interval)
            field = optics.circular_aperture(field)
            field = optics.propagate_fresnel(field,
                                             distance=distance,
                                             input_sample_interval=input_sample_interval,
                                             wave_lengths=wave_lengths)
            return field

        optical_system = optics.OpticalSystem(forward_model,
                                              wave_resolution=wave_resolution,
                                              wave_lengths=wave_lengths,
                                              sensor_resolution=(512,512),
                                              discretization_size=input_sample_interval,
                                              use_planar_incidence=False,
                                              depth_bins=depth_bins)

        sensor_img_graph = optical_system.get_sensor_img(input_img=input_img,
                                                   depth_dependent=True,
                                                   depth_map=test_depth)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sensor_img = sess.run(sensor_img_graph)

    sensor_img = sensor_img.squeeze()
    plt.imshow(sensor_img)
    plt.show()

    #for i in range(blurred_imgs.shape[-1]):
    #    plt.imshow(blurred_imgs[:,:,:,i].squeeze())
    #    plt.show()


def test_fftshift():
    tf.reset_default_graph()

    input_img = cv2.imread('test_imgs/Lenna.png')
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float64)
    input_img /= 255.0
    input_img = np.mean(input_img, 2)[None, :, :, None]

    np_fft = np.fft.fftshift( np.fft.fft2( input_img.squeeze() ))
    np_ifft = np.fft.ifft2( np.fft.ifftshift( np_fft )).astype(np.float64)

    input_img_tf = tf.cast(input_img, tf.complex64)
    objFT = optics.transp_fft2d( input_img_tf)
    tf_fft_graph = optics.fftshift2d_tf( tf.transpose(objFT, [0,2,3,1]))
    tf_ifft_graph = optics.transp_ifft2d( tf.transpose( optics.ifftshift2d_tf(tf_fft_graph), [0,3,1,2]))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        tf_fft, tf_ifft = sess.run([tf_fft_graph, tf_ifft_graph])

    tf_fft = tf_fft.squeeze().astype(np.float64)
    tf_ifft = tf_ifft.squeeze().astype(np.float64)

    print(np.amax(np.abs(np_fft - tf_fft)))
    print(np.amax(np.abs(np_ifft - tf_ifft)))

    plt.imshow(np_ifft)
    plt.show()

    plt.imshow(tf_ifft)
    plt.show()

    plt.imshow(np.log(np.abs(np_fft)).astype(np.float64))
    plt.show()

    plt.imshow(np.log(np.abs(tf_fft)).astype(np.float64))
    plt.show()

    plt.imshow(np.log(np.abs(np_fft - tf_fft) + 1e-9))
    plt.show()
    plt.imshow(np.log(np.abs(tf_ifft - np_ifft) + 1e-9))
    plt.show()


def test_area_downsampling():
    tf.reset_default_graph()

    input_img = cv2.imread('test_imgs/Lenna.png')
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float64)
    input_img /= 255.0
    input_img = np.mean(input_img, 2)

    input_side_length = input_img.shape[1]
    target_side_length = 341

    opencv_reference = cv2.resize(input_img,
                                  (target_side_length, target_side_length),
                                  interpolation=cv2.INTER_AREA)
    opencv_reference_bilinear = cv2.resize(input_img,
                                  (target_side_length, target_side_length),
                                  interpolation=cv2.INTER_CUBIC)

    input_img = input_img[None, :, :, None]
    input_img = tf.convert_to_tensor(input_img, tf.float64)

    target_img_graph = optics.area_downsampling_tf(input_img, target_side_length)
    reference_graph = tf.image.resize_images(input_img, (target_side_length, target_side_length))


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        target_img, reference = sess.run([target_img_graph, reference_graph])


    plt.imshow(target_img.squeeze())
    plt.show()
    plt.imshow(reference.squeeze())
    plt.show()
    fig, ax = plt.subplots()
    cax = ax.imshow((target_img - reference).squeeze())
    cbar = fig.colorbar(cax)
    plt.show()
    fig, ax = plt.subplots()
    cax = ax.imshow((target_img.squeeze() - opencv_reference).squeeze())
    cbar = fig.colorbar(cax)
    plt.show()
    fig, ax = plt.subplots()
    cax = ax.imshow((opencv_reference_bilinear - opencv_reference).squeeze())
    cbar = fig.colorbar(cax)
    plt.show()


def test_fft_conv2d():
    test_psf = np.square(np.random.randn(25,25))
    test_psf /= np.sum(test_psf)
    test_img = cv2_load_rgb("/home/sci/workspace/flatcam/src/test_imgs/Lenna.png")
    test_img_gray = np.mean(test_img, axis=2).squeeze()

    print(test_psf.shape)
    print(test_img_gray.shape)

    np_conv = scipy.ndimage.filters.convolve(test_img_gray, test_psf, mode='wrap')

    with tf.device('/device:GPU:2'):
        tf_fast_conv_graph = optics.fft_conv2d(test_img_gray[None,:,:,None],
                                                test_psf[:,:,None, None])

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        tf_fast_conv = sess.run(tf_fast_conv_graph)

    fig, ax = plt.subplots(1,2, figsize=(10,20))
    ax[0].imshow(np_conv)
    ax[1].imshow(tf_fast_conv.squeeze())
    plt.show()

    fig, ax = plt.subplots(figsize=(10,10))
    cax = ax.imshow((np_conv - tf_fast_conv.squeeze()).squeeze())
    cbar = fig.colorbar(cax)
    plt.show()

