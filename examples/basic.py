# coding=utf-8
import tensorflow as tf

logs_path = './basic_tmp'

# 建立一個 graph，並建立兩個常數 op ，這些 op 稱為節點
g1 = tf.Graph()
with g1.as_default():
    a = tf.constant([1.5, 6.0], name='a')
    b = tf.Variable([1.5, 3.2], name='b')
    c = (a * b) + (a / b)
    d = c * a
    y = tf.assign(b, d)

with tf.Graph().as_default() as g2:
    # 建立一個 1x2 矩陣與 2x1 矩陣 op
    m1 = tf.constant([[1., 0., 2.], [-1., 3., 1.]])
    m2 = tf.constant([[3., 1.], [2., 1.], [1., 0.]])
    m3 = tf.matmul(m1, m2) # 矩陣相乘

# 在 session 執行 graph，並進行資料數據操作 `c`。
# 然後指派給 cpu 做運算
with tf.Session(graph=g1) as sess_cpu:
  with tf.device("/cpu:0"):
      sess_cpu.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter(logs_path, graph=g1)
      print(sess_cpu.run(y))

with tf.Session(graph=g2) as sess_gpu:
  with tf.device("/gpu:0"):
      result = sess_gpu.run(m3)
      print(result)

# 使用 tf.InteractiveSession 方式來印出內容(不會實際執行)
it_sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始器 initializer op 的 run() 方法初始化 'x'
x.initializer.run()
sub = tf.subtract(x, a)

print sub.eval()
it_sess.close()
