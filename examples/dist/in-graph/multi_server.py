# coding=utf-8
import tensorflow as tf

tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# 定義 Cluster
cluster = tf.train.ClusterSpec({
    "worker": ["localhost:2222", "localhost:2223", "localhost:2224"]
})

# 建立 Worker server
server = tf.train.Server(cluster, job_name="worker", task_index=FLAGS.task_index)
server.join()

# python server.py --task_index=0
# python server.py --task_index=1
