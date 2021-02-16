import tensorflow as tf
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
tensor = tf.ones([8], dtype=tf.int32)
updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
print(updated)
