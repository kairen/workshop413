# coding=utf-8
import tensorflow as tf

tf.app.flags.DEFINE_string("worker_hosts", "", "list of hostname:port pairs")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# 定義 Cluster
worker_hosts = FLAGS.worker_hosts.split(",")
cluster = tf.train.ClusterSpec({"worker": worker_hosts})

# 建立 Worker server
server = tf.train.Server(cluster, job_name="worker", task_index=FLAGS.task_index)
server.join()

#  python multi_server.py --worker_hosts="localhost:2222,localhost:2223,localhost:2224" --task_index=0
