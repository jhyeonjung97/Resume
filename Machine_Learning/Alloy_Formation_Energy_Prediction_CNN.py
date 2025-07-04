import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import time
tic = time.time()

d_train = np.loadtxt("formation-energy-train.csv", delimiter=",", dtype=np.float32)
x_train = d_train[:, :5]
y_train = d_train[:, -1:]
mean = np.mean(x_train, 0)
std = np.std(x_train, 0)
x_train = (x_train - mean) / std

d_dev = np.loadtxt("formation-energy-dev.csv", delimiter=",", dtype=np.float32)
x_dev = d_dev[:, :5]
y_dev = d_dev[:, -1:]
x_dev = (x_dev - mean) / std

d_test = np.loadtxt("formation-energy-test.csv", delimiter=",", dtype=np.float32)
x_test = d_test[:, :5]
y_test = d_test[:, -1:]
x_test = (x_test - mean) / std

X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal([5, 5]))
b1 = tf.Variable(tf.zeros([5]))
a1 = tf.nn.leaky_relu(tf.matmul(X, w1) + b1)
#a1 = tf.nn.dropout(a1, 0.9)
w2 = tf.Variable(tf.random_normal([5, 7]))
b2 = tf.Variable(tf.zeros([7]))
a2 = tf.nn.leaky_relu(tf.matmul(a1, w2) + b2)
w3 = tf.Variable(tf.random_normal([7, 1]))
b3 = tf.Variable(tf.zeros([1]))
P = tf.matmul(a2, w3) + b3

cost = tf.reduce_mean(tf.square(P-Y))
#reg1 = tf.scalar_mul(0.1, tf.reduce_mean(tf.square(w1)))
#cost = tf.add_n([cost, reg1, reg2])
opt = tf.train.GradientDescentOptimizer(0.02).minimize(cost)
error = tf.reduce_mean(np.abs((P-Y)/Y))*100

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(6000):
    sess.run(opt, feed_dict={X: x_train, Y: y_train})
        if (step + 1) % 1000 == 0:
            print(step + 1, sess.run(cost, feed_dict={X: x_train, Y: y_train}))
            print("dev_error:", sess.run(error, feed_dict={X: x_dev, Y: y_dev}), '%')

p_test = sess.run(P, feed_dict={X: x_test, Y: y_test})
np.savetxt("test.csv", p_test, fmt='%.3f', delimiter=',')

sess.close()
toc = time.time()
print("time:", toc-tic)
