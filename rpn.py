import tensorflow as tf
import numpy as np
batch_size = 2
nms_threshold = 0.5
proposal_count = 10

def apply_delta_to_anchors(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [batch, N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:,:, 0] * height
    center_x += deltas[:,:, 1] * width
    height *= tf.exp(deltas[:,:, 2])
    width *= tf.exp(deltas[:,:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=-1)
    return result

def clip_boxes(boxes, window):
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

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes

def rpn_net(input_tensor, anchors_num, anchor_stride):
    shared = tf.layers.conv2d(input_tensor, 512, 3, padding = "same", activation = tf.nn.relu, strides = (anchor_stride,anchor_stride))
    x = tf.layers.conv2d(shared, 5*anchors_num, 1, padding = "same")
    return tf.reshape(x, (batch_size, -1, 5))

def rpn(input_tensor, anchors):
    '''
    returns Proposals in normalized coordinates [batch, proposal_count, (y1, x1, y2, x2)]
    '''
    x = rpn_net(input_tensor, 4, 1)
    scores = x[:,:,0]
    delta = x[:,:,1:]
    boxes = apply_delta_to_anchors(anchors, delta)
    batch_proposals = []
    window = np.array([0, 0, 1, 1], dtype=np.float32)
    for i in range(batch_size):
        box = clip_boxes(boxes[i], window)
        indices = tf.image.non_max_suppression(box, scores[i],  proposal_count, nms_threshold)
        proposals = tf.gather(box, indices)
        padding = tf.maximum(proposal_count - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        batch_proposals.append(proposals)
    proposals = tf.stack(batch_proposals)
    return proposals

shape = [8,8]
anchors = generate_anchors([2,2], [1,2], shape,1,1)

input_tensor= tf.placeholder(tf.float32, [None, None,None, 3])
net = rpn(input_tensor, anchors)

if __name__=="__main__":

	data = np.zeros((batch_size, shape[0], shape[1], 3))
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		x= sess.run(net, feed_dict = {input_tensor:data})
		print(x)