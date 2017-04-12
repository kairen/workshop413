import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 定義訓練資料
dataset = np.array([
    ((-0.4, 0.3), 0),
    ((-0.3, -0.1), 0),
    ((-0.2, 0.4), 0),
    ((-0.1, 0.1), 0),
    ((0.6, -0.5), 0),
    # 非線性 point
    ((0.8, 0.7), 1),
    ((0.9, -0.5), 1),
    ((0.7, -0.9), 1),
    ((0.8, 0.2), 1),
    ((0.4, -0.6), 1)
])

x_data = np.matrix([x for x,y in dataset])
y_data = np.matrix([y for x,y in dataset]).T

xs = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.zeros([1, 1]) + 1)
z = tf.matmul(xs, W) + b
o = tf.sigmoid(z)

# Logistic 專用的 cost function
cross_entropy = tf.reduce_mean(ys * -tf.log(o) + (1-ys) * -tf.log(1-o))
#cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(z, ys))

# 使用梯度下降法
train = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

optimal_W = None
optimal_b = None
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={xs:x_data, ys:y_data})
        if i % 50 == 0:
            print sess.run(cross_entropy, feed_dict={xs:x_data, ys:y_data})
    optimal_W = sess.run(W)
    optimal_b = sess.run(b)

    # 新增圖表
    ps = [v[0] for v in dataset]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter([v[0] for v in ps[:5]], [v[1] for v in ps[:5]], s=10, c='b', marker="o", label='O')
    ax1.scatter([v[0] for v in ps[5:]], [v[1] for v in ps[5:]], s=10, c='r', marker="x", label='X')
    l = np.linspace(-2,2)
    a,b = -optimal_W[0][0]/optimal_W[1][0], -optimal_b[0][0]/optimal_W[1][0]
    ax1.plot(l, a*l + b, 'b-')
    plt.legend(loc='upper left');
    plt.show()
