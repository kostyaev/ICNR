import numpy as np
import tensorflow as tf
from tensorflow.python.layers.utils import normalize_tuple

class ICNR:
    def __init__(self, initializer, scale=1):
        """ICNR initializer for checkerboard artifact free transpose convolution

        Code adapted from https://github.com/kostyaev/ICNR
        Discussed at https://github.com/Lasagne/Lasagne/issues/862
        Original paper: https://arxiv.org/pdf/1707.02937.pdf

        Parameters
        ----------
        initializer : Initializer
            Initializer used for kernels (glorot uniform, etc.)
        scale : iterable of two integers, or a single integer
            Stride of the transpose convolution
            (a.k.a. scale factor of sub pixel convolution)
        """
        self.scale = normalize_tuple(scale, 2, "scale")
        self.initializer = initializer

    def __call__(self, shape, dtype):
        if self.scale == 1:
            return self.initializer(shape)
        size = shape[:2]
        new_shape = np.array(shape)
        new_shape[:2] //= self.scale
        x = self.initializer(new_shape, dtype)
        x = tf.transpose(x, perm=[2, 0, 1, 3])
        x = tf.image.resize(x, size=size, method="nearest")
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        return x

