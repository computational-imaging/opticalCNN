import numpy as np
import math
import timeit

from datetime import datetime
import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

import layers.optics as optics
from layers.utils import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# test a model with various constraints
def train(params, summary_every=100, print_every=250, save_every=1000, verbose=True):
    # Unpack params
    isNonNeg = params.get('isNonNeg', False)
    # addBias = params.get('addBias', True)
    numIters = params.get('numIters', 1000)
    activation = params.get('activation', tf.nn.relu)

    # constraint helpers
    def nonneg(input_tensor):
        return tf.square(input_tensor) if isNonNeg else input_tensor

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    # input placeholders
    classes = 9
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, classes])
        keep_prob = tf.placeholder(tf.float32)

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        padamt = 28
        dim = 84       
        paddings = tf.constant([[0, 0,], [padamt, padamt], [padamt, padamt], [0, 0]])
        x_image = tf.pad(x_image, paddings)
        x_image = tf.image.resize_nearest_neighbor(x_image, size=(dim, dim))
        tf.summary.image('input', x_image, 3)

    # build model
    if True:
        doOpticalConv=True
        if doOpticalConv:
            doAmplitudeMask=False
            hm_reg_scale = 1e-2
            r_NA = 35
            
            # initialize with optimized phase mask
            mask = np.load('maskopt/opticalcorrelator_w-conv1_height-map-sqrt.npy')
            initializer = tf.constant_initializer(mask)
            # initializer=None
     
            h_conv1 = optical_conv_layer(x_image, hm_reg_scale, r_NA, n=1.48, wavelength=532e-9,
                       activation=activation, amplitude_mask=doAmplitudeMask, initializer=initializer,
                       name='opt_conv1')
            # h_conv2 = optical_conv_layer(h_conv1, hm_reg_scale, r_NA, n=1.48, wavelength=532e-9,
            #            activation=activation, amplitude_mask=doAmplitudeMask, name='opt_conv2')
        else:
            conv1dim = dim
            W_conv1 = weight_variable([conv1dim, conv1dim, 1, 1], name='W_conv1')
            W_conv1_flip = tf.reverse(W_conv1, axis=[0,1])
            # W_conv1 = weight_variable([12, 12, 1, 9])
            W_conv1_im = tf.expand_dims(tf.expand_dims(tf.squeeze(W_conv1), 0),3)
            optics.attach_summaries("W_conv1", W_conv1_im, image=True)
            h_conv1 = activation(conv2d(x_image, nonneg(W_conv1_flip)))
            
            # h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)

            # W_conv2 = weight_variable([48, 48, 1, 1], name='W_conv2')
            # W_conv2 = weight_variable([12, 12, 9, 9])
            # h_conv2 = activation(conv2d(h_conv1_drop, nonneg(W_conv2)))

        # h_conv1_split = tf.split(h_conv1, 9, axis=3)
        # h_conv1_tiled = tf.concat([tf.concat(h_conv1_split[:3], axis=1), 
        #                            tf.concat(h_conv1_split[3:6], axis=1), 
        #                            tf.concat(h_conv1_split[6:9], axis=1)], axis=2)
        # tf.summary.image("h_conv1", h_conv1_tiled, 3)
        
        # h_conv2_split = tf.split(h_conv2, 9, axis=3)
        # h_conv2_tiled = tf.concat([tf.concat(h_conv2_split[:3], axis=1), 
        #                            tf.concat(h_conv2_split[3:6], axis=1), 
        #                            tf.concat(h_conv2_split[6:9], axis=1)], axis=2)
        # tf.summary.image("h_conv2", h_conv2_tiled, 3)
        
        optics.attach_summaries("h_conv1", h_conv1, image=True)
        #optics.attach_summaries("h_conv2", h_conv2, image=True)
        # h_conv2 = x_image

        doFC = False
        if doFC:            
            with tf.name_scope('fc'):
                h_conv1 = tf.cast(h_conv1, dtype=tf.float32)
                fcsize = dim*dim
                W_fc1 = weight_variable([fcsize, classes], name='W_fc1')
                h_conv1_flat = tf.reshape(h_conv1, [-1, fcsize])
                y_out = (tf.matmul(h_conv1_flat, nonneg(W_fc1)))

                # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
                # W_fc2 = weight_variable([hidden_dim, 10])
                # y_out = tf.matmul(h_fc1_drop, nonneg(W_fc2))
        else:
            doConv2 = False
            if doConv2:
                if doOpticalConv:
                    h_conv2 = optical_conv_layer(h_conv1, hm_reg_scale, r_NA, n=1.48, wavelength=532e-9,
                           activation=activation, name='opt_conv2')
                    h_conv2 = tf.cast(h_conv2, dtype=tf.float32)
                else:
                    W_conv2 = weight_variable([dim, dim, 1, 1])
                    W_conv2_flip = tf.reverse(W_conv2, axis=[0,1])
                    W_conv2_im = tf.expand_dims(tf.expand_dims(tf.squeeze(W_conv2), 0),3)
                    optics.attach_summaries("W_conv2", W_conv2_im, image=True)
                    h_conv2 = activation(conv2d(h_conv1, nonneg(W_conv2_flip)))
                    
                    W_conv3 = weight_variable([dim, dim, 1, 1])
                    W_conv3_flip = tf.reverse(W_conv3, axis=[0,1])
                    W_conv3_im = tf.expand_dims(tf.expand_dims(tf.squeeze(W_conv3), 0),3)
                    optics.attach_summaries("W_conv3", W_conv3_im, image=True)
                    h_conv3 = activation(conv2d(h_conv2, nonneg(W_conv3_flip)))
                    
                    
                tf.summary.image("h_conv2", h_conv2)
                tf.summary.image("h_conv3", h_conv3)
                split_1d = tf.split(h_conv3, num_or_size_splits=3, axis=1)
            else:
                split_1d = tf.split(h_conv1, num_or_size_splits=3, axis=1)
            
            h_conv_split = tf.concat([tf.split(split_1d[0], num_or_size_splits=3, axis=2),
                                       tf.split(split_1d[1], num_or_size_splits=3, axis=2),
                                       tf.split(split_1d[2], num_or_size_splits=3, axis=2)], 0)
            # h_conv2_split1, h_conv2_split2 = tf.split(h_conv2, num_or_size_splits=2, axis=1)
            # h_conv2_split = tf.concat([tf.split(h_conv2_split1, num_or_size_splits=5, axis=2),
            #                            tf.split(h_conv2_split2, num_or_size_splits=5, axis=2)], 0)
            y_out = tf.transpose(tf.reduce_max(h_conv_split, axis=[2,3,4]))

        tf.summary.image('output', tf.reshape(y_out, [-1, 3, 3, 1]), 3)

    # loss, train, acc
    with tf.name_scope('cross_entropy'):
        total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out)
        mean_loss = tf.reduce_mean(total_loss)
        tf.summary.scalar('loss', mean_loss)

    # train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(mean_loss)
    train_step = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, rho=1.0).minimize(mean_loss)
    
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

    def get_feed(train):
        if train:
            x, y = mnist.train.next_batch(50)
        else:
            x = mnist.test.images
            y = mnist.test.labels
            
        # remove "0"s
        indices = ~np.equal(y[:,0], 1)
        x_filt = np.squeeze(x[indices])
        y_filt = np.squeeze(y[indices,1:])
        
        return x_filt, y_filt
    
    x_test, y_test = get_feed(train=False)
    
    for i in range(FLAGS.num_iters):
        x_train, y_train = get_feed(train=True)
        _, loss, train_accuracy, train_summary = sess.run([train_step, mean_loss, accuracy, merged], feed_dict=
                           {x: x_train, y_: y_train, keep_prob: FLAGS.dropout})
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
                print('step %d: loss %g, train acc %g' %
                      (i, loss, train_accuracy))

    # test_acc = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
    # print('final step %d, train accuracy %g, test accuracy %g' %
    #       (i, train_accuracy, test_acc))
    #sess.close()

    train_writer.close()
    test_writer.close()

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # try different constraints
    params = {}
    params['isNonNeg'] = True
    params['activation'] = tf.identity

    train(params, summary_every=200, print_every=50, save_every=1000, verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default=8001,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
    now = datetime.now()
    run_id = now.strftime('%Y%m%d-%H%M%S')
    # run_id = 'optconv/'
    parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join('checkpoints/mnist/', run_id),
      help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
