import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from datetime import datetime
import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow as tf


import layers.optics as optics
import layers.optics_alt as optics_alt
from layers.utils import *
from layers.data_utils import get_CIFAR10_grayscale

# test a model with various constraints
def train(params, summary_every=100, print_every=250, save_every=1000, verbose=True):
    # Unpack params
    wavelength = params.get('wavelength', 532e-9)
    isNonNeg = params.get('isNonNeg', False)
    numIters = params.get('numIters', 1000)
    activation = params.get('activation', tf.nn.relu)
    opt_type = params.get('opt_type', 'ADAM')
    
    # switches
    doMultichannelConv = params.get('doMultichannelConv', False)
    doMean = params.get('doMean', False)
    doOpticalConv = params.get('doOpticalConv', True)
    doAmplitudeMask = params.get('doAmplitudeMask', False)
    doZernike =  params.get('doZernike', False)
    doFC =  params.get('doFC', False)
    doConv1 =  params.get('doConv1', True)
    doConv2 =  params.get('doConv2', True)
    doConv3 =  params.get('doConv3', False)
    doNonnegReg =  params.get('doNonnegReg', False)
    doOptNeg =  params.get('doOptNeg', False)
    doTiledConv =  params.get('doTiledConv', False)

    z_modes = params.get('z_modes', 1024)
    convdim1 =  params.get('convdim1', 100)
    convdim2 =  params.get('convdim2', 100)
    convdim3 =  params.get('convdim3', 100)
    
    depth1 =  params.get('depth1', 3)
    depth2 =  params.get('depth2', 3)
    depth3 =  params.get('depth3', 3)
    
    padamt = params.get('padamt', 0)
    dim = params.get('dim', 60) 
    
    buff = params.get('buff', 4)
    rows = params.get('rows', 4)
    cols = params.get('cols', 4)

    # constraint helpers
    def nonneg(input_tensor):
        return tf.abs(input_tensor) if isNonNeg else input_tensor
    
    def vis_weights(W_conv, depth, buff, rows, cols, name):
        kernel_list = tf.split(tf.transpose(W_conv, [2, 0, 1, 3]), depth, axis=3)
        kernels_pad = [tf.pad(kernel, [[0,0], [buff, buff], [buff+4, buff+4], [0,0]]) 
                       for kernel in kernel_list]
        W_conv_tiled = tf.concat([tf.concat(kernels_pad[i*cols:(i+1)*cols], axis=2) for i in range(rows)], axis=1)
        tf.summary.image(name, W_conv_tiled, 3)
        
    def vis_h(h_conv, depth, rows, cols, name):
        # this was for viewing multichannel convolution
        h_conv_split = tf.split(h_conv, depth, axis=3)
        h_conv_tiled = tf.concat([tf.concat(h_conv_split[i*cols:(i+1)*cols], axis=2) for i in range(rows)], axis=1)
        tf.summary.image(name, h_conv_tiled, 3)

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    # input placeholders
    classes = 10
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 32, 32])
        y_ = tf.placeholder(tf.int64, shape=[None])
        keep_prob = tf.placeholder(tf.float32)

        x_image = tf.reshape(x, [-1, 32, 32, 1])
        paddings = tf.constant([[0, 0,], [padamt, padamt], [padamt, padamt], [0, 0]])
        x_image = tf.pad(x_image, paddings)
        # x_image = tf.image.resize_nearest_neighbor(x_image, size=(dim, dim))
        tf.summary.image('input', x_image, 3)
        
        # if not isNonNeg and not doNonnegReg:
        #     x_image -= tf.reduce_mean(x_image)
            
    # regularizers
    global_step = tf.Variable(0, trainable=False)
    if doNonnegReg:
        reg_scale = tf.train.polynomial_decay(0.,
                                              global_step,
                                              decay_steps=6000,
                                              end_learning_rate=6000.)
        psf_reg = optics_alt.nonneg_regularizer(reg_scale)
    else:
        psf_reg = None
        
    l2_reg = tf.contrib.layers.l2_regularizer(1e-1 , scope=None)

    # build model
    h_conv_out = x_image
    fcdepth = 1
    doVis = True
    
    if doConv1:
        with tf.name_scope('conv1'):
            if doTiledConv:
                tiled_dim = (32)*rows
                init_vals_pos = tf.truncated_normal([tiled_dim, tiled_dim, 1, 1], stddev=0.1) + .1
                W_conv1_tiled = tf.Variable(init_vals_pos, name='W_conv1_tiled')
                W_conv1_tiled = nonneg(W_conv1_tiled)
                tf.summary.image("W_conv1_tiled", tf.expand_dims(tf.squeeze(W_conv1_tiled, -1), 0))
                
                tile_pad = tiled_dim//2 - 16
                tile_paddings = tf.constant([[0, 0,], [tile_pad, tile_pad], [tile_pad, tile_pad], [0, 0]])
                x_padded = tf.pad(x_image, tile_paddings)
                tf.summary.image('input', x_padded, 3)
                
                fftpadamt = int(tiled_dim/2)
                h_conv_tiled = tf.abs(optics.fft_conv2d(fftpad(x_padded, fftpadamt), fftpad_psf(W_conv1_tiled, fftpadamt)))
                h_conv_tiled = fftunpad(tf.cast(h_conv_tiled, dtype=tf.float32), fftpadamt)
                
                h_conv_split2d = split2d_layer(h_conv_tiled, rows, cols)
                b_conv1 = bias_variable([depth1], 'b_conv1')
                h_conv1 = h_conv_split2d + b_conv1
            elif doOpticalConv:
                tiled_dim = (32)*cols
                tile_pad = tiled_dim//2 - 16
                tile_paddings = tf.constant([[0, 0,], [tile_pad, tile_pad], [tile_pad, tile_pad], [0, 0]])
                x_padded = tf.pad(x_image, tile_paddings)
                tf.summary.image('input', x_padded, 3)
               
                r_NA = tiled_dim/2
                hm_reg_scale = 1e-2
                # initialize with optimized phase mask
                # mask = np.load('maskopt/quickdraw9_zernike1024.npy')
                # initializer = tf.constant_initializer(mask)
                initializer=None
                
                h_conv1_opt = optical_conv_layer(x_padded, hm_reg_scale, r_NA, n=1.48, wavelength=wavelength,
                       activation=None, amplitude_mask=doAmplitudeMask, zernike=doZernike, 
                       n_modes=z_modes, initializer=initializer, name='opt_conv1_pos')
                
                # h_conv1_opt_neg = optical_conv_layer(x_padded, hm_reg_scale, r_NA, n=1.48, wavelength=wavelength,
                #        activation=None, amplitude_mask=doAmplitudeMask, zernike=doZernike, 
                #        n_modes=z_modes, initializer=initializer, name='opt_conv1_neg')
                
                h_conv1_opt = tf.cast(h_conv1_opt, dtype=tf.float32)
                h_conv_split2d = split2d_layer(h_conv1_opt, 2*rows, cols)
                b_conv1 = bias_variable([depth1], 'b_conv1')
                h_conv1 = h_conv_split2d + b_conv1
                
            else:        
                if doOptNeg:
                    # positive weights
                    init_vals_pos = tf.truncated_normal([convdim1, convdim1, 1, depth1], stddev=0.1) + .1
                    W_conv1_pos = tf.Variable(init_vals_pos, name='W_conv1_pos')
                    # W_conv1 = weight_variable([convdim1, convdim1, 1, depth1], name='W_conv1')
                    W_conv1_pos = nonneg(W_conv1_pos)
                    #W_conv1_nonneg /= tf.reduce_sum(tf.abs(W_conv1_nonneg)) # conservation of energy
                    tf.contrib.layers.apply_regularization(l2_reg, weights_list=[tf.transpose(W_conv1_pos, [3,0,1,2])])
                   
                    # negative weights
                    init_vals_neg = tf.truncated_normal([convdim1, convdim1, 1, depth1], stddev=0.1) +.1
                    W_conv1_neg = tf.Variable(init_vals_neg, name='W_conv1_neg')
                    # W_conv1 = weight_variable([convdim1, convdim1, 1, depth1], name='W_conv1')
                    W_conv1_neg = nonneg(W_conv1_neg)
                    # W_conv1_nonneg /= tf.reduce_sum(tf.abs(W_conv1_nonneg)) # conservation of energy
                    tf.contrib.layers.apply_regularization(l2_reg, weights_list=[tf.transpose(W_conv1_neg, [3,0,1,2])])
                   
                    W_conv1 = tf.subtract(W_conv1_pos, W_conv1_neg)

                    if doVis:
                        vis_weights(W_conv1_pos, depth1, buff, rows, cols, 'W_conv1_pos')
                        vis_weights(W_conv1_neg, depth1, buff, rows, cols, 'W_conv1_neg')

                elif isNonNeg:
                    init_vals = tf.truncated_normal([convdim1, convdim1, 1, depth1], stddev=0.1)
                    W_conv1 = tf.Variable(init_vals, name='W_conv1_nn')+.1
                    # W_conv1 = weight_variable([convdim1, convdim1, 1, depth1], name='W_conv1')
                    W_conv1 = nonneg(W_conv1)
                    #W_conv1_nonneg /= tf.reduce_sum(tf.abs(W_conv1_nonneg)) # conservation of energy
                else:
                    W_conv1 = weight_variable([convdim1, convdim1, 1, depth1], name='W_conv1')

                    if psf_reg is not None:
                        tf.contrib.layers.apply_regularization(psf_reg, weights_list=[tf.transpose(W_conv1, [3,0,1,2])])

                vis_weights(W_conv1, depth1, buff, rows, cols, 'W_conv1')
                
                W_conv1_flip = tf.reverse(W_conv1, axis=[0,1]) # flip if using tfconv
                h_conv1 = conv2d(x_image, W_conv1_flip)
                h_conv1 /= tf.reduce_max(h_conv1, axis=[1,2,3], keep_dims=True)
                
                b_conv1 = bias_variable([depth1], 'b_conv1') 
                h_conv1 = h_conv1 + b_conv1


            vis_h(h_conv1, depth1, rows, cols, 'h_conv1')
            variable_summaries("h_conv1", h_conv1)
            h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)
            
            #h_pool1 = max_pool_2x2(h_conv1)
            h_pool1 = h_conv1_drop
            
            if doNonnegReg:
                h_pool1 = optics_alt.shifted_relu(h_pool1)
            else:
                h_pool1 = activation(h_pool1)
            variable_summaries("h_conv1_post", h_pool1)
            
            h_conv_out = h_pool1
            #dim = 16
            fcdepth = depth1
                    
    if doConv2:
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([convdim2, convdim2, depth1, depth2], name='W_conv2')
            # vis_weights(W_conv2, depth2, buff, rows, cols, 'W_conv2')
            b_conv2 = bias_variable([depth2], name='b_conv2')
            h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
            
            # h_pool2 = max_pool_2x2(h_conv2)
            h_pool2 = h_conv2
            variable_summaries("h_conv2", h_pool2)
            
            h_conv2_drop = tf.nn.dropout(h_pool2, keep_prob)
            h_conv2_drop = activation(h_conv2_drop)
            variable_summaries("h_conv2_post", h_conv2_drop)
            h_conv_out = h_conv2_drop
            # dim = 16
            fcdepth = depth2
        
    if doConv3:
        with tf.name_scope('conv3'):
            W_conv3 = weight_variable([convdim3, convdim3, depth2, depth3], name='W_conv3')
            # vis_weights(W_conv3, depth3, buff, rows, cols, 'W_conv3')
            b_conv3 = bias_variable([depth3], name='b_conv3')
          
            h_conv3 = conv2d(h_pool2, W_conv3) + b_conv3
            h_pool3 = max_pool_2x2(h_conv3)
            variable_summaries("h_conv3", h_pool3)
            
            h_conv3_drop = tf.nn.dropout(h_pool3, keep_prob)
            h_conv3_drop = activation(h_conv3_drop)
            variable_summaries("h_conv3_post", h_conv3_drop)
            h_conv_out = h_conv3_drop
            fcdepth = depth3
            dim = 16

    # choose output layer here
    with tf.name_scope('fc'):
        h_conv_out = tf.cast(h_conv_out, dtype=tf.float32)
        
        fcsize = dim*dim*fcdepth
        hidden_dim = classes
        W_fc1 = weight_variable([fcsize, hidden_dim], name='W_fc1')
        b_fc1 = bias_variable([hidden_dim], name='b_fc1')
        h_conv_flat = tf.reshape(h_conv_out, [-1, fcsize])

        y_out = tf.matmul(h_conv_flat, W_fc1) + b_fc1

        # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        # W_fc2 = weight_variable([hidden_dim, classes])
        # b_fc2 = bias_variable([classes])
        # y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            
    tf.summary.image('output', tf.reshape(y_out, [-1, 2, 5, 1]), 3)

    # loss, train, acc
    with tf.name_scope('cross_entropy'):
        total_data_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y_, classes), logits=y_out)
        data_loss = tf.reduce_mean(total_data_loss)
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = tf.add(data_loss, reg_loss)
        tf.summary.scalar('data_loss', data_loss)
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('total_loss', total_loss)

    if opt_type == 'ADAM':
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss, global_step)
    elif opt_type == 'adadelta':
        train_step = tf.train.AdadeltaOptimizer(FLAGS.learning_rate_ad, rho=.9).minimize(total_loss, global_step)
    else:
        train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum=0.5, use_nesterov=True).minimize(total_loss, global_step)
    
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_out, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    losses = []

    # tensorboard setup
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    tf.global_variables_initializer().run()
    
    # add ops to save and restore all the variables
    saver = tf.train.Saver(max_to_keep=2)
    save_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
    
    x_train_all, y_train_all, x_test, y_test, _, _ = get_CIFAR10_grayscale(num_training=49000, num_validation=1000, num_test=0)
    num_training = x_train_all.shape[0]
    def get_feed(train, batch_size=50, augmentation=False):
        idcs = np.random.randint(0, num_training, batch_size)
        x = x_train_all[idcs, :, :]
        y = y_train_all[idcs]
        
        if augmentation:
            angle = np.random.uniform(low=0.0, high=20.0)
            x = rotate(x, angle, axes=(2,1), reshape=True)
            x = resize(x, (32,32))
        
        return x, y
    
    for i in range(FLAGS.num_iters):
        x_train, y_train = get_feed(train=True, augmentation=False)
        _, loss, reg_loss_graph, train_accuracy, train_summary = sess.run(
                          [train_step, total_loss, reg_loss, accuracy, merged], 
                          feed_dict={x: x_train, y_: y_train, keep_prob: FLAGS.dropout})
        losses.append(loss)

        if i % summary_every == 0:
            train_writer.add_summary(train_summary, i)
            
            test_summary, test_accuracy = sess.run([merged, accuracy],
                                                   feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
            test_writer.add_summary(test_summary, i)
            if verbose:
                print('step %d: test acc %g' % (i, test_accuracy))
            
        if i > 0 and i % save_every == 0:
            # print("Saving model...")
            saver.save(sess, save_path, global_step=i)
            
            
            
        if i % print_every == 0:
            if verbose:
                print('step %d:\t loss %g,\t reg_loss %g,\t train acc %g' %
                      (i, loss, reg_loss_graph, train_accuracy))

    #test_batches = []
    # for i in range(4):
    #     idx = i*500
    #     batch_acc = accuracy.eval(feed_dict={x: x_test[idx:idx+500, :], y_: y_test[idx:idx+500], keep_prob: 1.0})
    #     test_batches.append(batch_acc)
    # test_acc = np.mean(test_batches)
    
    test_acc = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
    print('final step %d, train accuracy %g, test accuracy %g' %
          (i, train_accuracy, test_acc))
    #sess.close()

    train_writer.close()
    test_writer.close()

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # try different constraints
    params = {}
    params['wavelength'] = 532e-9
    params['activation'] = tf.nn.relu
    params['opt_type'] = 'ADAM'
    
    #params['doMultichannelConv'] = True
    params['doTiledConv'] = False
    params['doOpticalConv'] = True
    params['doAmplitudeMask'] = False
    params['doZernike'] = False
    params['doFC'] = True
    
    params['isNonNeg'] = True
    params['doOptNeg'] = True
    params['doNonnegReg'] = False
    
    params['doConv1'] = True
    params['doConv2'] = False
    params['doConv3'] = False
    
    params['convdim1'] = 9
    params['convdim2'] = 5
    params['convdim3'] = 3
    params['z_modes'] = 1024
    
    params['depth1'] = 8
    params['depth2'] = 16
    params['depth3'] = 16
    
    params['padamt'] = 0
    params['dim'] = 32
    
    params['buff'] = 6
    params['rows'] = 2
    params['cols'] = 4
    

    train(params, summary_every=200, print_every=100, save_every=1000, verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default=10001,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                      help='Initial learning rate')
    parser.add_argument('--learning_rate_ad', type=float, default=1,
                      help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Keep probability for training dropout.')  
    
    now = datetime.now()
    runtime = now.strftime('%Y%m%d-%H%M%S')
    
    run_id = 'endtoend/' + runtime + '/'#testing_2' 
    parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join('/media/data/checkpoints/onn/hybrid_cifar10/', run_id),
      help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
