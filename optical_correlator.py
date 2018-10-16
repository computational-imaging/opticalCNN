import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

from datetime import datetime
import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf

import layers.optics as optics
from layers.utils import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# test a model with various constraints
def train(params, summary_every=100, save_every=2000, verbose=True):
    # Unpack params
    isNonNeg = params.get('isNonNeg', False)
    addBias = params.get('addBias', True)
    doLogTrans = params.get('logtrans', False)
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
        padamt = 28 # can change to 0 for fully connected
        dim = 28+2*padamt
        paddings = tf.constant([[0, 0,], [padamt, padamt], [padamt, padamt], [0, 0]])
        x_image = tf.pad(x_image, paddings)
        tf.summary.image('input', x_image, 3)

    # build model
    doOpticalConv=False # optimize opt-conv layer?
    doConv=True # conv layer or fully connected layer?
    if doConv:
        if doOpticalConv:
            doAmplitudeMask=False # amplitude or phase mask?
            hm_reg_scale = 1e-2
            r_NA = 35 # numerical aperture radius of mask, in  pixels
            h_conv1 = optical_conv_layer(x_image, hm_reg_scale, r_NA, n=1.48, wavelength=532e-9,
                       activation=activation, amplitude_mask=doAmplitudeMask, name='opt_conv1')
            h_conv1 = tf.cast(h_conv1, dtype=tf.float32)
        else:
            W_conv1 = weight_variable([dim, dim, 1, 1], name='W_conv1')
            W_conv1 = nonneg(W_conv1)
            W_conv1_im = tf.expand_dims(tf.expand_dims(tf.squeeze(W_conv1), 0),3)
            optics.attach_summaries("W_conv1", W_conv1_im, image=True)

            # W_conv1 = weight_variable([12, 12, 1, 9])
            h_conv1 = activation(conv2d(x_image, (W_conv1)))            

        optics.attach_summaries("h_conv1", h_conv1, image=True)
        h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)

        # h_conv1_split = tf.split(h_conv1, 9, axis=3)
        # h_conv1_tiled = tf.concat([tf.concat(h_conv1_split[:3], axis=1), 
        #                            tf.concat(h_conv1_split[3:6], axis=1), 
        #                            tf.concat(h_conv1_split[6:9], axis=1)], axis=2)
        # tf.summary.image("h_conv1", h_conv1_tiled, 3)

        split_1d = tf.split(h_conv1_drop, num_or_size_splits=3, axis=1)
        h_conv1_split = tf.concat([tf.split(split_1d[0], num_or_size_splits=3, axis=2),
                                   tf.split(split_1d[1], num_or_size_splits=3, axis=2),
                                   tf.split(split_1d[2], num_or_size_splits=3, axis=2)], 0)
        y_out = tf.transpose(tf.reduce_max(h_conv1_split, axis=[2,3,4]))

    else:
        # single fully connected layer instead, for comparison
        with tf.name_scope('fc'):
            fcsize = dim*dim
            W_fc1 = weight_variable([fcsize, classes], name='W_fc1')
            W_fc1 = nonneg(W_fc1)

            # visualize the FC weights
            W_fc1_split = tf.reshape(tf.transpose(W_fc1), [classes, 28, 28])
            W_fc1_split = tf.split(W_fc1_split, classes, axis=0)
            W_fc1_tiled = tf.concat([tf.concat(W_fc1_split[:3], axis=2),
                                     tf.concat(W_fc1_split[3:6], axis=2),
                                     tf.concat(W_fc1_split[6:9], axis=2)], axis=1)
            tf.summary.image("W_fc1", tf.expand_dims(W_fc1_tiled, 3))


            h_conv1_flat = tf.reshape(x_image, [-1, fcsize])
            y_out = (tf.matmul(h_conv1_flat, (W_fc1)))

    tf.summary.image('output', tf.reshape(y_out, [-1, 3, 3, 1]), 3)

    # loss, train, acc
    with tf.name_scope('cross_entropy'):
        total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out)
        mean_loss = tf.reduce_mean(total_loss)
        tf.summary.scalar('loss', mean_loss)

    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(mean_loss)

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
    
    # MNIST feed dict
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
    
    # QuickDraw feed dict
    # train_data = np.load('/media/data/Datasets/quickdraw/split/all_train.npy')
    # test_data = np.load('/media/data/Datasets/quickdraw/split/all_test.npy')
    # def get_feed(train, batch_size=50):
    #     if train:
    #         idcs = np.random.randint(0, np.shape(train_data)[0], batch_size)
    #         x = train_data[idcs, :]
            
    #         categories = idcs//4000
    #         y = np.zeros((batch_size, classes))
    #         y[np.arange(batch_size), categories] = 1
          
    #    else:
    #        x = test_data 
    #        y = np.resize(np.equal(range(classes),0).astype(int),(100,classes))
    #        for i in range(1,classes):
    #            y = np.concatenate((y, np.resize(np.equal(range(classes),i).astype(int),(100,classes))), axis=0)
        
    #    return x, y
            
    x_test, y_test = get_feed(train=False)
    for i in range(FLAGS.num_iters):
        x_train, y_train = get_feed(train=True)
        _, loss = sess.run([train_step, mean_loss], feed_dict={x: x_train, y_: y_train, keep_prob: FLAGS.dropout})
        losses.append(loss)

        if i % summary_every == 0:
            train_summary, train_accuracy = sess.run([merged, accuracy],
                                                     feed_dict={
              x: x_train, y_: y_train, keep_prob: FLAGS.dropout})
            train_writer.add_summary(train_summary, i)
            
            test_summary, test_accuracy = sess.run([merged, accuracy],
                                                   feed_dict={
              x: x_test, y_: y_test, keep_prob: 1.0})
            # test_writer.add_summary(test_summary, i)
            
            if verbose:
                print('step %d: loss %g, train acc %g, test acc %g' %
                      (i, loss, train_accuracy, test_accuracy))
                
        if i % save_every == 0:
            print("Saving model...")
            saver.save(sess, save_path, global_step=i)

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
    now = datetime.now()
    params['isNonNeg'] = True
    params['activation'] = tf.identity

    train(params, summary_every=100, verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default=10000,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
    now = datetime.now()
    run_id = now.strftime('%Y%m%d-%H%M%S')
    parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join('checkpoints/correlator/', run_id),
      help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
