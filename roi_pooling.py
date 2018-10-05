import tensorflow as tf
import numpy as np

def Roi_pooling(input_tensor, output_shape = [4, 4]):
	shape = tf.shape(input_tensor)
	box = tf.cast([[0, 0, 1, 1],[0.5,0.5,1,1]], tf.float32)
	return  tf.image.crop_and_resize(input_tensor, boxes = box, box_ind = [0, 0], crop_size = output_shape, method="bilinear")

input_tensor= tf.placeholder(tf.float32)
net = Roi_pooling(input_tensor)
net2 = tf.image.resize_bilinear(input_tensor, [2,2], align_corners = True)

a = np.ones((1, 8, 8, 1))
a[0,5,5,0] = 8
a[0,4,4,0] = 9
a[0,7,7,0] = 4
a[0,6,7,0] = 6
a[0,3,2,0] = 11

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	x= sess.run(net, feed_dict = {input_tensor:a})
	print(x[0])
	print("==========")
	print(x[1])
	# print(x2)