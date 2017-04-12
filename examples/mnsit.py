# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# reset everything to rerun in jupyter
tf.reset_default_graph()

# config
batch_size = 100
learning_rate = 0.5
training_epochs = 5
logs_path = "./tmp_mnist/1"

# load mnist data set
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# input images
with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
    # target 10 output classes
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

# model parameters will change during training so we use tf.Variable
with tf.name_scope("weights"):
    W = tf.Variable(tf.zeros([784, 10]))

# bias
with tf.name_scope("biases"):
    b = tf.Variable(tf.zeros([10]))

# implement model
with tf.name_scope("softmax"):
    # y is our prediction
    y = tf.nn.softmax(tf.matmul(x,W) + b)

# specify cost function
with tf.name_scope('cross_entropy'):
    # this is our cost
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# specify optimizer
with tf.name_scope('train'):
    # optimizer is an "operation" which we can execute in a session
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create a summary for our cost and accuracy
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)

# merge all summaries into a single "operation" which we can execute in a session
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # variables need to be initialized before we can use them
    sess.run(tf.global_variables_initializer())

    # create log writer object
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # perform training cycles
    for epoch in range(training_epochs):

        # number of batches in one epoch
        batch_count = int(mnist.train.num_examples/batch_size)

        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # perform the operations we defined earlier on batch
            _, summary = sess.run([train_op, summary_op], feed_dict={x: batch_x, y_: batch_y})

            # write log
            writer.add_summary(summary, epoch * batch_count + i)

        if epoch % 5 == 0:
            print "Epoch: ", epoch
    print "Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print "done"

# tensorboard --logdir=run1:./tmp_mnist/1 --port=6006
