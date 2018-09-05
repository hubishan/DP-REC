#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Title     : TODO
# @Objective : TODO
# @Time      : 2018/9/5 22:50
# @Author    : hubishan
# @Site      :
# @File      : mnist.py
# @Software  : IntelliJ IDEA

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# import input_data

print ("packs loaded")
print ("Download and Extract MNIST dataset")
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

print (" tpye of 'mnist' :  %s" % (type(mnist)))
print (" number of trian data  :  %d" % (mnist.train.num_examples))
print (" number of test data  :  %d" % (mnist.test.num_examples))
# What does the data of MNIST look like?
print ("What does the data of MNIST look like?")
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print()
print (" type of 'trainimg'  :  %s"    % (type(trainimg)))
print (" type of 'trainlabel'  :  %s"  % (type(trainlabel)))
print (" type of 'testimg'  :  %s"     % (type(testimg)))
print (" type of 'testlabel'  :  %s"   % (type(testlabel)))
print (" shape of 'trainimg'  :  %s"   % (trainimg.shape,))
print (" shape of 'trainlabel'  :  %s" % (trainlabel.shape,))
print (" shape of 'testimg'  :  %s"    % (testimg.shape,))
print (" shape of 'testlabel'  :  %s"  % (testlabel.shape,))

nsample = 5
randidx = np.random.randint(trainimg.shape[0], size=nsample)

for i in randidx:
    curr_img   = np.reshape(trainimg[i, :], (28, 28)) # 28 by 28 matrix
    curr_label = np.argmax(trainlabel[i, :] ) # Label

    # plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.matshow(curr_img)

    plt.title("" + str(i) + "th Training Data "
              + "Label is " + str(curr_label))
    print ("" + str(i) + "th Training Data "
           + "Label is " + str(curr_label))
    plt.show()
    # plt.figure()
    # plt.plot(curr_img)



