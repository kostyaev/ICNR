from __future__ import division

import tensorflow as tf


class ICNR:
    """ICNR initializer for checkerboard artifact free sub pixel convolution

    Ref:
     [1] Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
     https://arxiv.org/pdf/1707.02937.pdf)

    Args:
    initializer: initializer used for sub kernels (orthogonal, glorot uniform, etc.)
    scale: scale factor of sub pixel convolution
    """

    def __init__(self, initializer, scale=1):
        self.scale = scale
        self.initializer = initializer

    def __call__(self, shape, dtype):
        shape = list(shape)
        if self.scale == 1:
            return self.initializer(shape)

        new_shape = shape[:3] + [shape[3] // (self.scale ** 2)]
        x = self.initializer(new_shape, dtype)
        x = tf.transpose(x, perm=[2, 0, 1, 3])
        size = (shape[0] * self.scale, shape[1] * self.scale)
        x = tf.image.resize(x, size=size, method="nearest")
        x = tf.nn.space_to_depth(x, block_size=self.scale)
        x = tf.transpose(x, perm=[1, 2, 0, 3])

        return x
