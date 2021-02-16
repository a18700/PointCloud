import tensorflow as tf
import numpy as np

# batch of 2 inputs of 13x13 pixels with 3 channels each.
# Four 5x5 filters applied to each channel, so 12 total channels output
inputs_np = np.ones((2, 13, 13, 3))
inputs = tf.constant(inputs_np)
# Build the filters so that their behavior is easier to understand.  For these filters
# which are 5x5, I set the middle pixel (location 2,2) to some value and leave
# the rest of the pixels at zero
filters_np = np.zeros((5,5,3,4)) # 5x5 filters for 3 inputs and applying 4 such filters to each one.
filters_np[2, 2, 0, 0] = 2.0
filters_np[2, 2, 0, 1] = 2.1
filters_np[2, 2, 0, 2] = 2.2
filters_np[2, 2, 0, 3] = 2.3
filters_np[2, 2, 1, 0] = 3.0
filters_np[2, 2, 1, 1] = 3.1
filters_np[2, 2, 1, 2] = 3.2
filters_np[2, 2, 1, 3] = 3.3
filters_np[2, 2, 2, 0] = 4.0
filters_np[2, 2, 2, 1] = 4.1
filters_np[2, 2, 2, 2] = 4.2
filters_np[2, 2, 2, 3] = 4.3
filters = tf.constant(filters_np)

out = tf.nn.depthwise_conv2d(
      inputs,
      filters,
      strides=[1,1,1,1],
      padding='SAME')

out_shape = tf.shape(out)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out_val = out.eval()
    out_val_shape = out_shape.eval()

print("output {}".format(out_val))
print("output shape {}".format(out_val.shape)) # 2, 13, 13, 12
print("output cases 0 and 1 identical? {}".format(np.all(out_val[0]==out_val[1])))
print("One of the pixels for each of the 12 output {} ".format(out_val[0, 6, 6]))
# Output:
# output cases 0 and 1 identical? True
# One of the pixels for each of the 12 output [ 2.   2.1  2.2  2.3  3.   3.1  3.2  3.3  4.   4.1  4.2  4.3]


