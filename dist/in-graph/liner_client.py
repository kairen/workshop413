# coding=utf-8
import tensorflow as tf
import numpy as np

# 執行目標 Session
server_target = "grpc://localhost:2222"
logs_path = './liner_tmp'

train_X = np.random.rand(100).astype(np.float32)
train_Y = train_X * 0.1 + 0.3

# 指定 worker task 0 使用 CPU 運算
with tf.device("/job:worker/task:0"):
    with tf.device("/cpu:0"):
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)
        w = tf.Variable(0.0, name="weight")
        b = tf.Variable(0.0, name="bias")
        y = w * X + b

        loss = tf.reduce_mean(tf.square(y - Y))

        init_op = tf.global_variables_initializer()
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 使用 Master session target
with tf.Session(server_target) as sess:
    sess.run(init_op)
    for i in range(500):
        sess.run(train_op, feed_dict={X: train_Y, Y: train_Y})
        if i % 50 == 0:
            print i, sess.run(w), sess.run(b)

    print "\nWeight: {0}, Bias: {1}".format(sess.run(w), sess.run(b))

    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
