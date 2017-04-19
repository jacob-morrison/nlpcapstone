# Jacob Morrison

import tensorflow as tf
from tensorflow.python.ops import rnn
import numpy as np
import data_helpers
import os
import time
import datetime

# parameters
learning_rate = 0.0001
training_iters = 500000
batch_size = 64
display_step = 10

# network parameters
n_words = 75 # truncate sentences (pad sentences with <PAD> tokens if less than this, cut off if larger)
n_dim = 300
n_hidden = 300 # hidden layer num of features
n_classes = 16 # 15 total senses

# tf graph input
x1 = tf.placeholder(tf.float32, [None, n_dim, None])
x2 = tf.placeholder(tf.float32, [None, n_dim, None])
x1_len = tf.placeholder(tf.float32, [None])
x2_len = tf.placeholder(tf.float32, [None])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden*2, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, cell):
    # split input into n_words [batch_size, n_dim] tensors
    x = tf.unstack(x, n_words, 2)

    # get the output of the cell
    outputs, states = rnn.rnn(cell, x, dtype=tf.float32)
    #outputs, states = rnn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, sequence_length=lengths, time_major=True)

    # return last output from cell
    return outputs[-1]

# this returns all outputs, which is wrong. need to find the last relevant output (aka at seqlen)
def Dynamic_RNN(x, cell, lengths):
    # get output of the cell
    outputs, _ = rnn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, sequence_length=lengths, time_major=False)

    return outputs[-1]

def BiRNN(x, f_cell, b_cell):

    x = tf.unstack(x, n_words, 2)

    outputs, _, _ = rnn.bidirectional_rnn(f_cell, b_cell, x, dtype=tf.float32)

    return outputs[-1]

# define an LSTM cell
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

# forwards cell
lstm_cell_forwards = tf.nn.rnn_cell.BasicLSTMCell(n_hidden/2, forget_bias=1.0)

# backwards cell
lstm_cell_backwards = tf.nn.rnn_cell.BasicLSTMCell(n_hidden/2, forget_bias=1.0)

# need to get a prediction for each sentence

# get the vector representation of each word
with tf.variable_scope('scope1') as scope1:
    #pred1 = Dynamic_RNN(tf.transpose(x1, perm=[0, 2, 1]), lstm_cell, x1_len)#, [-1, n_hidden])
    pred1 = RNN(x1, lstm_cell)
    #pred1 = BiRNN(x1, lstm_cell_forwards, lstm_cell_backwards)
with tf.variable_scope('scope1') as scope1:
    scope1.reuse_variables()
    #pred2 = Dynamic_RNN(tf.transpose(x2, perm=[0, 2, 1]), lstm_cell, x2_len)#, [-1, n_hidden])
    pred2 = RNN(x2, lstm_cell)
    #pred2 = BiRNN(x2, lstm_cell_forwards, lstm_cell_backwards)

# do something with both representations
# simple concatenation?
out = tf.tanh(tf.concat(1, [pred1, pred2]))

# predict the relation class
pred = tf.add(tf.matmul(out, weights['out']), biases['out'])

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initializing all variables
init = tf.global_variables_initializer()

# launch the graph
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
tf.add_to_collection('accuracy', accuracy)
tf.add_to_collection('x1', x1)
tf.add_to_collection('x2', x2)
tf.add_to_collection('y', y)

with tf.Session() as sess:
    sess.run(init)
    step = 1
    model = data_helpers.load_model('./Data/GoogleNews-vectors-negative300.bin')
    sentences1, sentences2, labels, lengths1, lengths2 = \
                data_helpers.load_labels_and_data_PDTB(model, './Data/PDTB_implicit/train.txt', False, True, True)
    total = 0

    while total < training_iters:
        start = total  % len(sentences1)
        end = (total + batch_size) % len(sentences1)
        if end <= start:
            end = len(sentences1)
        batch_x1 = sentences1[start : end]
        batch_x2 = sentences2[start : end]
        batch_x1_lengths = lengths1[start : end]
        batch_x2_lengths = lengths2[start : end]
        batch_y = labels[start : end]
        total += (len(batch_x1))
        sess.run(optimizer, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, x1_len: batch_x1_lengths, x2_len: batch_x2_lengths})
        if step % display_step == 0:
            #calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], \
                            feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, x1_len: batch_x1_lengths, x2_len: batch_x2_lengths})
            print "Iter " + str(total) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc)
            # extract variables here
            #w2 = sess.run(weights['w2'])
            #print(w2)
        step += 1
    print "Training finished!"

    # calculate training set accuracy
    print("testing accuracy on training set: ")
    step = 0
    acc = 0.
    print(len(sentences1))
    batch_size2 = batch_size * 2
    while step * batch_size2 < len(sentences1):
        start = (step * batch_size2)
        end = ((step + 1) * batch_size2)
        if end > len(sentences1):
            end = len(sentences1)
        batch_x1 = sentences1[start : end]
        batch_x2 = sentences2[start : end]
        batch_x1_lengths = lengths1[start : end]
        batch_x2_lengths = lengths2[start : end]
        batch_y = labels[start : end]
        acc += (float(len(batch_x1)) / len(sentences1)) * \
                            sess.run(accuracy, feed_dict={x1: batch_x1, x2: batch_x2, \
                                y: batch_y, x1_len: batch_x1_lengths, x2_len: batch_x2_lengths})
        step += 1
    print(str(acc))

    # test accuracy on dev set
    print("accuracy on dev set:")
    sentences12, sentences22, labels2, lengths12, lengths22 = data_helpers.load_labels_and_data_PDTB(\
        model, \
        './Data/PDTB_implicit/dev.txt', \
        False, \
        True, True)                          
    print(str(sess.run(accuracy, feed_dict={x1: sentences12, x2: sentences22, y: labels2, x1_len: lengths12, x2_len: lengths22})))

