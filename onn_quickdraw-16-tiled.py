import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import functools

from datetime import datetime
import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow as tf

import layers.optics as optics
import layers.optics_alt as optics_alt
from layers.utils import *


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
    doNonnegReg =  params.get('doNonnegReg', False)

    z_modes = params.get('z_modes', 1024)
    convdim1 =  params.get('convdim1', 100)
    
    classes = 16
    cdim1 = params.get('cdim1', classes)
    
    padamt = params.get('padamt', 0)
    dim = params.get('dim', 60) 
    
    tiling_factor = params.get('tiling_factor', 5)
    tile_size = params.get('tile_size', 56)
    kernel_size = params.get('kernel_size', 7) 

    # constraint helpers
    def nonneg(input_tensor):
        return tf.square(input_tensor) if isNonNeg else input_tensor

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    # input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.int64, shape=[None, classes])
        keep_prob = tf.placeholder(tf.float32)

        x_image = tf.reshape(x, [-1, 28, 28, 1]) 
        paddings = tf.constant([[0, 0,], [padamt, padamt], [padamt, padamt], [0, 0]])
        x_image = tf.pad(x_image, paddings)
        # x_image = tf.image.resize_nearest_neighbor(x_image, size=(dim, dim))
        tf.summary.image('input', x_image, 3)
        
        # if not isNonNeg and not doNonnegReg:
        #     x_image -= tf.reduce_mean(x_image)

    # nonneg regularizer
    global_step = tf.Variable(0, trainable=False)
    if doNonnegReg:
        reg_scale = tf.train.polynomial_decay(0.,
                                              global_step,
                                              decay_steps=8000,
                                              end_learning_rate=10000.)
        psf_reg = optics_alt.nonneg_regularizer(reg_scale)
    else:
        psf_reg = None
    
    # build model 
    # single tiled convolutional layer
    h_conv1 = optics_alt.tiled_conv_layer(x_image, tiling_factor, tile_size, kernel_size, 
                                          name='h_conv1', nonneg=isNonNeg, regularizer=psf_reg)
    optics.attach_img("h_conv1", h_conv1)
    
    split_1d = tf.split(h_conv1, num_or_size_splits=4, axis=1)

    # calculating output scores
    h_conv_split = tf.concat([tf.split(split_1d[0], num_or_size_splits=4, axis=2),
                              tf.split(split_1d[1], num_or_size_splits=4, axis=2),
                              tf.split(split_1d[2], num_or_size_splits=4, axis=2),
                              tf.split(split_1d[3], num_or_size_splits=4, axis=2)], 0)
    if doMean:
        y_out = tf.transpose(tf.reduce_mean(h_conv_split, axis=[2,3,4]))
    else:
        y_out = tf.transpose(tf.reduce_max(h_conv_split, axis=[2,3,4]))

    tf.summary.image('output', tf.reshape(y_out, [-1, 4, 4, 1]), 3)

    # loss, train, acc
    with tf.name_scope('cross_entropy'):
        total_data_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out)
        data_loss = tf.reduce_mean(total_data_loss)
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = tf.add(data_loss, reg_loss)
        tf.summary.scalar('data_loss', data_loss)
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('total_loss', total_loss)

    if opt_type == 'ADAM':
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss, global_step)
    elif opt_type == 'Adadelta':
        train_step = tf.train.AdadeltaOptimizer(FLAGS.learning_rate_ad, rho=.9).minimize(total_loss, global_step)
    else:
        train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum=0.5, use_nesterov=True).minimize(total_loss, global_step)
    
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
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

    # change to your directory
    train_data = np.load('/media/data/Datasets/quickdraw/split/quickdraw16_train.npy')
    test_data = np.load('/media/data/Datasets/quickdraw/split/quickdraw16_test.npy')
    def get_feed(train, batch_size=50):
        if train:
            idcs = np.random.randint(0, np.shape(train_data)[0], batch_size)
            x = train_data[idcs, :]
            y = np.zeros((batch_size, classes))
            y[np.arange(batch_size), idcs//8000] = 1
            
        else:
            x = test_data
            y = np.zeros((np.shape(test_data)[0], classes))
            y[np.arange(np.shape(test_data)[0]), np.arange(np.shape(test_data)[0])//100] = 1                
        
        return x, y
    
    x_test, y_test = get_feed(train=False)
    
    for i in range(FLAGS.num_iters):
        x_train, y_train = get_feed(train=True)
        _, loss, reg_loss_graph, train_accuracy, train_summary = sess.run(
                          [train_step, total_loss, reg_loss, accuracy, merged], 
                          feed_dict={x: x_train, y_: y_train, keep_prob: FLAGS.dropout})
        losses.append(loss)

        if i % summary_every == 0:
            train_writer.add_summary(train_summary, i)
            
        if i > 0 and i % save_every == 0:
            # print("Saving model...")
            saver.save(sess, save_path, global_step=i)
            
            # test_summary, test_accuracy = sess.run([merged, accuracy],
            #                                        feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
            # test_writer.add_summary(test_summary, i)
            # if verbose:
            #     print('step %d: test acc %g' % (i, test_accuracy))
            
        if i % print_every == 0:
            if verbose:
                print('step %d:\t loss %g,\t reg_loss %g,\t train acc %g' %
                      (i, loss, reg_loss_graph, train_accuracy))
                
    
    test_batches = []
    for i in range(32):
        idx = i*50
        batch_acc = accuracy.eval(feed_dict={x: x_test[idx:idx+50, :], y_: y_test[idx:idx+50, :], keep_prob: 1.0})
        test_batches.append(batch_acc)
    test_acc = np.mean(test_batches)   
    
    #test_acc = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
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
    params['isNonNeg'] = True
    params['activation'] = tf.identity # functools.partial(shifted_relu, thresh=10.0)
    params['opt_type'] = 'ADAM'
    
    params['doMultichannelConv'] = False
    params['doMean'] = False
    params['doOpticalConv'] = False
    
    params['doNonnegReg'] = False
    
    params['padamt'] = 64
    params['dim'] = 40*4
    
    params['tiling_factor'] = 4
    params['tile_size'] = 40
    params['kernel_size'] = 32

    train(params, summary_every=10, print_every=10, save_every=1000, verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default=10001,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                      help='Initial learning rate')
    parser.add_argument('--learning_rate_ad', type=float, default=1,
                      help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')  
    now = datetime.now()
    runtime = now.strftime('%Y%m%d-%H%M%S')
    run_id = 'quickdraw_tiled_nonneg/' + runtime + '/'
    parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join('checkpoints/', run_id),
      help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
