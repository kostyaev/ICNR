# Checkerboard artifact free sub pixel convolution initialization
Tensorflow implementation of ICNR initialization used in https://arxiv.org/pdf/1707.02937.pdf

![screenshot](kernel_vis.png)

Updated for TF v2.4.

## Usage example:

Wrap up your initialization with ICRN and you are ready to go:
```python
layers.Conv2DTranspose(..., strides=scale, kernel_initializer=ICNR(GlorotUniform(), scale))

```
