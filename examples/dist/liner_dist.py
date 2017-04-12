# coding=utf-8
import tensorflow as tf
import numpy as np

# 輸入的參數 flag
tf.app.flags.DEFINE_string("ps_hosts", "", "list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("log_path", "", "Log path")
tf.app.flags.DEFINE_integer("loss_value", 100, "loss value")
FLAGS = tf.app.flags.FLAGS


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    print(ps_hosts, worker_hosts)
    # 建立定義 Cluster and server
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)
    print("Cluster Job Name: %s, Task index: %d, target: %s" % (FLAGS.job_name, FLAGS.task_index, server.target))

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        train_X = np.linspace(-1.0, 1.0, 100)
        train_Y = 2.0 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10.0

        X = tf.placeholder("float")
        Y = tf.placeholder("float")

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # 建立 linear regression model ...
            w = tf.Variable(0.0, name="weight")
            b = tf.Variable(0.0, name="bias")

            # 損失函式，用於描述模型預測值與真實值的差距大小，常見為`均方差(Mean Squared Error)`
            loss = tf.square(Y - tf.multiply(X, w) - b)

            global_step = tf.Variable(0, name="global_step")

            train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        # 建立 "Supervisor" 來負責監督訓練過程，管理各個 Process
        # is_chief: 指定哪個 task index 進行初始化工作
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=FLAGS.log_path,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # managed_session: Async 模式，變數初始化就開始工作.
        # prepare_or_wait_for_session: Sync 模式，要完成變數初始化跟主節點準備好才工作.
        with sv.managed_session(server.target) as sess:
            loss_value = 200
            while not sv.should_stop() and loss_value > FLAGS.loss_value:
                # 執行一個非同步 training 步驟.
                # 若要執行同步可利用`tf.train.SyncReplicasOptimizer` 來進行
                for (x, y) in zip(train_X, train_Y):
                    _, step = sess.run([train_op, global_step],
                                       feed_dict={X: x, Y: y})

                loss_value = sess.run(loss, feed_dict={X: x, Y: y})
                print("Step: {}, loss: {}".format(step, loss_value))

        # 結束後停止 supervisor
        sv.stop()


if __name__ == "__main__":
    tf.app.run()

'''
python liner_dist.py --ps_hosts="localhost:2222" --worker_hosts="localhost:2223,localhost:2224" --job_name=worker --task_index=0 --log_path=./tmp/wk0 --loss_value=35
'''
