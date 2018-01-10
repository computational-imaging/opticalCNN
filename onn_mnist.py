import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

from datetime import datetime
import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf

import layers.optics as optics
from layers.utils import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# test a model with various constraints 
def train(params, summary_every=50, print_every=250, verbose=True):
    # Unpack params
    isNonNeg = params.get('isNonNeg', False)
    addBias = params.get('addBias', True)
    doLogTrans = params.get('logtrans', False)
    log_dir = params.get('log_dir','checkpoints/')
    numIters = params.get('numIters', 1000)
    
    # constraint helpers
    def nonneg(input_tensor):
        return tf.square(input_tensor) if isNonNeg else input_tensor
    
    # log transmission
    a = tf.Variable(.5, [1])
    tf.summary.scalar('log-a', tf.abs(a))
    b = tf.Variable(.1, [1])
    tf.summary.scalar('log-b', b)
    def activation(v):
        if doLogTrans:
            transmittance = tf.add(tf.multiply(tf.abs(a), (tf.abs(v)+.05)), b)
            transmittance = tf.clip_by_value(transmittance, 0.1, 1)
            return tf.multiply(v,transmittance)
        else:
            return v
    
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    # input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        keep_prob = tf.placeholder(tf.float32)
        
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        padamt = 21
        dim = 28+2*padamt
        paddings = tf.constant([[0, 0,], [padamt, padamt], [padamt, padamt], [0, 0]])
        x_image = tf.pad(x_image, paddings)
        tf.summary.image('input', x_image, 3)

    # build model
    with tf.device('/device:GPU:2'):
        
        doOpticalConv=True
        if doOpticalConv:
            hm_reg_scale = 1e-2
            r_NA = 20
            h_conv1 = optical_conv_layer(x_image, hm_reg_scale, r_NA, n=1.48, wavelength=532e-9,
                       activation=None, name='opt_conv1')
            h_conv1 = tf.cast(h_conv1, dtype=tf.float32)
        else:
            W_conv1 = weight_variable([48, 48, 1, 1])
            h_conv1 = tf.nn.relu(conv2d(x_image, nonneg(W_conv1)))
        
        tf.summary.image("h_conv1", h_conv1)

        W_conv2 = weight_variable([48, 48, 1, 1])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, nonneg(W_conv2)))
        tf.summary.image("h_conv2", h_conv1)
        
        doGAP = False
        if doGAP:
            W_conv3 = weight_variable([48, 48, 1, 1])
            h_conv3 = tf.nn.relu(conv2d(h_conv2, nonneg(W_conv3)))
            tf.summary.image("h_conv3", h_conv2)
            # h_conv3_split = tf.split(h_conv3, num_or_size_splits=10, axis=1)
            h_conv3_split1, h_conv3_split2 = tf.split(h_conv3, num_or_size_splits=2, axis=1)
            h_conv3_split = tf.concat([tf.split(h_conv3, num_or_size_splits=5, axis=2), 
                                       tf.split(h_conv3, num_or_size_splits=5, axis=2)], 0)
            y_out = tf.transpose(tf.reduce_mean(h_conv3_split, axis=[2,3,4]))
        else:
            hidden_dim = 10
            W_fc1 = weight_variable([dim*dim, hidden_dim])
            h_pool2_flat = tf.reshape(h_conv2, [-1, dim*dim])
            y_out = tf.nn.relu(tf.matmul(h_pool2_flat, nonneg(W_fc1)))
        
            # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # W_fc2 = weight_variable([hidden_dim, 10])
            # y_out = tf.matmul(h_fc1_drop, nonneg(W_fc2))
    
        tf.summary.image('output', tf.reshape(y_out, [-1, 2, 5, 1]), 3)
        
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
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    
    tf.global_variables_initializer().run()
    
    for i in range(FLAGS.num_iters):
        batch = mnist.train.next_batch(50)
        _, loss = sess.run([train_step, mean_loss], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.9})
        losses.append(loss)
        
        if i % summary_every == 0:      
            train_summary, train_accuracy = sess.run([merged, accuracy], 
                                                     feed_dict={
              x: batch[0], y_: batch[1], keep_prob: 1.0}) 
            train_writer.add_summary(train_summary, i)
            test_summary, test_accuracy = sess.run([merged, accuracy], 
                                                   feed_dict={
              x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            test_writer.add_summary(test_summary, i)
            
        if i % print_every == 0:
            if verbose:
                print('step %d: loss %g, train acc %g, test acc %g' % 
                      (i, loss, train_accuracy, test_accuracy)) 

    test_acc = accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
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
    params['log_dir'] = FLAGS.log_dir
    params['isNonNeg'] = True

    train(params, summary_every=100, print_every=200, verbose=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default=10000,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                      help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
    now = datetime.now()
    # run_id = now.strftime('%Y%m%d-%H%M%S')
    run_id = 'optconv-testing'
    parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join('checkpoints/mnist_with_summaries/', run_id),
      help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
