#! usr/bin/env python3

import tensorflow as tf


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

net = tf.add(a, b)

sess = tf.Session()

binding = {a: 3.0, b: 4.5}
result = sess.run(net, feed_dict=binding)

sess.close()

print(result)

# python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'
# -c Directly runs the command that is being `man python`