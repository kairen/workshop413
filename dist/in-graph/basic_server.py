# coding=utf-8
import tensorflow as tf

# 定義 Cluster
cluster = tf.train.ClusterSpec({"worker": ["localhost:2222"]})

# 建立 Worker server
server = tf.train.Server(cluster,job_name="worker",task_index=0)
server.join()
