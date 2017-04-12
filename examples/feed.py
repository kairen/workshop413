# coding=utf-8
import tensorflow as tf

# 填充填充函式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
  # 透過 feed 來更改 op 內容，這只會在執行時有效
  print sess.run([output], feed_dict={input1:[7.], input2:[2.]})
