# coding=utf-8
import tensorflow as tf

# 建立一個變數 counter，並初始化為 0
state = tf.Variable(0, name="counter")

# 建立一個常數 op 為 1，並用來累加 state
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 啟動 Graph 前，變數必須先被初始化(init) op
init_op = tf.global_variables_initializer()

# 啟動 Graph 來執行 op
with tf.Session() as sess:
  sess.run(init_op)
  print sess.run(state)

  # 執行 op 並更新 state
  for _ in range(3):
    sess.run(update)
    print sess.run(state)
