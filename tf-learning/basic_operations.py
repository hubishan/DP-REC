
import tensorflow as tf
import os
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

print('TensorFlow version:{0}'.format(tf.__version__))
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

a=tf.constant(2)
b=tf.constant(3)

with tf.Session() as sess:
    print("a=2,b=3")
    print("常量节点相加：%i" % sess.run(a+b))
    print("常量节点相加：%i" % sess.run(a*b))


a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)

# 定义一些操作
add=tf.add(a,b)
mul=tf.multiply(a,b)


# 启动默认会话
with tf.Session() as sess:
    print("变量节点相加：%i" % sess.run(add, feed_dict={a: 2,b:3}))
    print("变量节点相加：%i" % sess.run(mul, feed_dict={a: 2,b:3}))


# 矩阵相乘（Matrix Multiplication）
# 闯将一个constant op，产生1*2的matirx

matrix1=tf.constant([[3,3]])

matrix2=tf.constant([[2],[2]])

# sess=tf.Session()
# print(sess.run(matrix1))
# print(sess.run(matrix2))
# [[3 3]]
# [[2]
#  [2]]

product =tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print("matrix multiply:",result)


writer = tf.summary.FileWriter(logdir='logs2',graph=tf.get_default_graph())
writer.flush()













