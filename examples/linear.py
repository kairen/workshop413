# coding=utf-8
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

# 用 numpy 亂數產生 100 個點資料
train_X = np.linspace(-1.0, 1.0, 100)
train_Y = 2.0 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10.0

X = tf.placeholder("float")
Y = tf.placeholder("float")

# 建立 linear regression model ...
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

# 損失函式，用於描述模型預測值與真實值的差距大小，常見為`均方差(Mean Squared Error)`
loss = tf.square(Y - tf.multiply(X, w) - b)

global_step = tf.Variable(0, name="global_step")
train_op = tf.train.AdagradOptimizer(0.01).minimize(
    loss, global_step=global_step)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

# Fit the line.
loss_value = 150
while loss_value > 90:
    for (x, y) in zip(train_X, train_Y):
        _, step = sess.run([train_op, global_step], feed_dict={X: x, Y: y})

    loss_value = sess.run(loss, feed_dict={X: x, Y: y})
    print("Step: {}, loss: {}, ".format(step, loss_value))
    # plt.plot(x_data, y_data, 'ro', label='Original data')
    # plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()

sess.close()
