# coding=utf-8
import tensorflow as tf
import sys
import numpy as np

batch_size = 20
input_dim = 10
logs_path = './ig_tmp'

# 推理函式
def inference(input, target):
    n_in = 10
    n_hide = 100
    n_out = 1
    W_1 = tf.get_variable(name='W_1', shape=(n_in, n_hide), initializer=tf.random_normal_initializer())
    b_1 = tf.get_variable(name='b_1', shape=(n_hide), initializer=tf.random_normal_initializer())
    W_2 = tf.get_variable(name='W_2', shape=(n_hide, n_out), initializer=tf.random_normal_initializer())
    b_2 = tf.get_variable(name='b_2', shape=(n_out), initializer=tf.random_normal_initializer())
    outputs_1 = tf.nn.relu(tf.matmul(input, W_1) + b_1)
    outputs_2 = tf.matmul(outputs_1, W_2) + b_2
    outputs_2 = tf.reshape(outputs_2, [-1])
    target = tf.reshape(target, [-1])
    loss = tf.reduce_sum(tf.square(target - outputs_2))
    return loss

# 平均梯度
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

# 取得 Train data
def get_tarin_data(file_path):
    one_batch_train_data = []
    one_batch_train_target = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            items = line.split()
            assert len(items) == input_dim + 1
            temp_data = [0] * input_dim
            for i in xrange(input_dim):
                temp_data[i] = float(items[i])
            one_batch_train_data.append(temp_data)
            one_batch_train_target.append(float(items[input_dim]))
            if len(one_batch_train_data) == batch_size * 2:
                one_batch_train_data = np.array(one_batch_train_data)
                x1 = np.array(one_batch_train_data[0 : batch_size, : ])
                y1 = np.array(one_batch_train_target[0 : batch_size])
                x2 = np.array(one_batch_train_data[batch_size : 2 * batch_size , : ])
                y2 = np.array(one_batch_train_target[batch_size : 2 * batch_size])
                yield (x1, y1, x2, y2)
                one_batch_train_data = []
                one_batch_train_target = []


def train_multi_device_model():

    cluster = tf.train.ClusterSpec({
        "worker": ["localhost:2222", "localhost:2223", "localhost:2224"]
    })

    server = tf.train.Server(cluster, job_name="worker", task_index=2)

    with tf.Session("grpc://localhost:2224") as sess:
        input1 = tf.placeholder(dtype = tf.float32, shape = [batch_size, input_dim])
        target1 = tf.placeholder(dtype = tf.float32, shape = [batch_size])
        input2 = tf.placeholder(dtype = tf.float32, shape = [batch_size, input_dim])
        target2 = tf.placeholder(dtype = tf.float32, shape = [batch_size])

        opt = tf.train.GradientDescentOptimizer(0.005)
        tower_grads = []

        with tf.device("/job:worker/task:0"):
            with tf.name_scope("work_1") as scope:
                loss1 = inference(input1, target1)
                grads1 = opt.compute_gradients(loss1)
                tower_grads.append(grads1)

        with tf.device("/job:worker/task:1"):
            with tf.name_scope("work_2") as scope:
                tf.get_variable_scope().reuse_variables()
                loss2 = inference(input2, target2)
                grads2 = opt.compute_gradients(loss2)
                tower_grads.append(grads2)

        grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads)

        init = tf.global_variables_initializer()
        sess.run(init)
        train_step = 0
        for x1, y1, x2, y2 in get_tarin_data("./train-data.txt"):
            feed_dict = {input1 : x1, target1 : y1, input2 : x2, target2 : y2}
            train_step += 1
            _loss1, _loss2, _ = sess.run([loss1, loss2 ,apply_gradient_op], feed_dict = feed_dict)
            assert not np.isnan(_loss1), 'Model diverged with loss = NaN'
            print "train_step : %d, loss1 : %f, loss2 : %f" % (train_step, _loss1, _loss2)

        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES
    train_multi_device_model()
