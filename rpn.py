import tensorflow as tf
batch_size = 2
max_output_size = 10
iou_threshold = 0.5
def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped

def rpn(input_tensor, anchors):
	shape = tf.shape(input_tensor)
	Xs = []
	for anchor in anchors:
		x = tf.layers.conv2d(input_tensor, 5, anchor, padding = "same")
		Xs.append(x)
	x = tf.concat(Xs, axis = -2)
	shape = tf.shape(x)
	scores = tf.reshape(x[:,:,:,0], [shape[0], shape[1]*shape[2], 1])
	boxes = tf.reshape(x[:,:,:,1:], [shape[0], shape[1]*shape[2], 4])
	batch_proposals = []
	window = np.array([0, 0, 1, 1], dtype=np.float32)
	for i in range(batch_size):
		indices = tf.images.non_max_supression(scores[i], clip_boxes_graph(boxes[i], window), max_output_size, iou_threshold)
		proposals = tf.gather(boxes[i], indices)
		padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
		proposals = tf.pad(proposals, [(0, padding), (0, 0)])
		batch_proposals.append(proposals)
	proposals = tf.stack(batch_proposals)

	return proposals
