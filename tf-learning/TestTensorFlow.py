
# from geetest import GeetestLib
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

print('TensorFlow version:{0}'.format(tf.__version__))
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

num_points=1000
vector_set=[]
for i in range(num_points):
    x1=np.random.normal(0.0,0.55)
    y1=0.1 * x1 + 0.3 + np.random.normal(0.0,0.03)
    vector_set.append([x1,y1])

x_train=[v[0] for v in vector_set]
y_train=[v[1] for v in vector_set]

# plt.scatter(x_train,y_train,c='r')
# plt.show()

# Model parameters
W = tf.Variable([0.1], dtype=tf.double,name='W')
b = tf.Variable([1], dtype=tf.double,name='b')

# Model input and output
x = tf.placeholder(tf.double)
linear_model = W*x + b
y=tf.placeholder(tf.double)


# loss
loss=tf.reduce_mean(tf.square(linear_model-y),name='loss')
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss,name='train')

# train data
# x_train=[1,2,3,4]
# y_train=[0,-1,-2,-3]

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for i in range(20):
    sess.run(train,{x:x_train,y:y_train})
    print('W=',sess.run(W),'b=',sess.run(b),'loss=',sess.run(loss,{x:x_train,y:y_train}))
    # sess.run(train)
    # print('W=',sess.run(W),'b=',sess.run(b),'loss=',sess.run(loss))

plt.scatter(x_train,y_train,c='r')
plt.plot(x_train,sess.run(W)*x_train+sess.run(b))
plt.show()

# print('----------')
# # evaluate training accuracy
# curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


# print(sess.run([W, b]))
