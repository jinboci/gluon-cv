# pylint: disable=arguments-differ
"""Bounding boxes operators"""
from __future__ import absolute_import

import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet import use_np
mx.npx.set_np()

@use_np
class NumPyBBoxCornerToCenter(object):
    """Convert corner boxes to center boxes using numpy.
    Corner boxes are encoded as (xmin, ymin, xmax, ymax)
    Center boxes are encoded as (center_x, center_y, width, height)

    Parameters
    ----------
    split : bool
        Whether split boxes to individual elements after processing.
    axis : int, default is -1
        Effective axis of the bounding box. Default is -1(the last dimension).

    Returns
    -------
     A BxNx4 NDArray if split is False, or 4 BxNx1 NDArray if split is True
    """

    def __init__(self, axis=-1, split=False):
        super(NumPyBBoxCornerToCenter, self).__init__()
        self._split = split
        self._axis = axis

    def __call__(self, x):
        xmin, ymin, xmax, ymax = np.split(x, 4, axis=self._axis)
        # note that we do not have +1 here since our nms and box iou does not.
        # this is different that detectron.
        width = xmax - xmin
        height = ymax - ymin
        x = xmin + width * 0.5
        y = ymin + height * 0.5
        if not self._split:
            return np.concatenate((x, y, width, height), axis=self._axis)
        else:
            return x, y, width, height


@use_np
class BBoxCornerToCenter(gluon.HybridBlock):
    """Convert corner boxes to center boxes.
    Corner boxes are encoded as (xmin, ymin, xmax, ymax)
    Center boxes are encoded as (center_x, center_y, width, height)

    Parameters
    ----------
    split : bool
        Whether split boxes to individual elements after processing.
    axis : int, default is -1
        Effective axis of the bounding box. Default is -1(the last dimension).

    Returns
    -------
     A BxNx4 NDArray if split is False, or 4 BxNx1 NDArray if split is True
    """

    def __init__(self, axis=-1, split=False):
        super(BBoxCornerToCenter, self).__init__()
        self._split = split
        self._axis = axis

    def forward(self, x):
        """Hybrid forward"""
        xmin, ymin, xmax, ymax = mx.np.split(x, axis=self._axis, num_outputs=4)
        # note that we do not have +1 here since our nms and box iou does not.
        # this is different that detectron.
        width = xmax - xmin
        height = ymax - ymin
        x = xmin + width * 0.5
        y = ymin + height * 0.5
        if not self._split:
            return mx.np.concatenate(x, y, width, height, axis=self._axis)
        else:
            return x, y, width, height


@use_np
class BBoxCenterToCorner(gluon.HybridBlock):
    """Convert center boxes to corner boxes.
    Corner boxes are encoded as (xmin, ymin, xmax, ymax)
    Center boxes are encoded as (center_x, center_y, width, height)

    Parameters
    ----------
    split : bool
        Whether split boxes to individual elements after processing.
    axis : int, default is -1
        Effective axis of the bounding box. Default is -1(the last dimension).

    Returns
    -------
     A BxNx4 NDArray if split is False, or 4 BxNx1 NDArray if split is True.
    """

    def __init__(self, axis=-1, split=False):
        super(BBoxCenterToCorner, self).__init__()
        self._split = split
        self._axis = axis

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        x, y, w, h = mx.np.split(x, axis=self._axis, num_outputs=4)
        hw = w * 0.5
        hh = h * 0.5
        xmin = x - hw
        ymin = y - hh
        xmax = x + hw
        ymax = y + hh
        if not self._split:
            return mx.np.concatenate(xmin, ymin, xmax, ymax, axis=self._axis)
        else:
            return xmin, ymin, xmax, ymax


@use_np
class BBoxSplit(gluon.HybridBlock):
    """Split bounding boxes into 4 columns.

    Parameters
    ----------
    axis : int, default is -1
        On which axis to split the bounding box. Default is -1(the last dimension).
    squeeze_axis : boolean, default is `False`
        If true, Removes the axis with length 1 from the shapes of the output arrays.
        **Note** that setting `squeeze_axis` to ``true`` removes axis with length 1 only
        along the `axis` which it is split.
        Also `squeeze_axis` can be set to ``true`` only if ``input.shape[axis] == num_outputs``.

    """

    def __init__(self, axis, squeeze_axis=False, **kwargs):
        super(BBoxSplit, self).__init__(**kwargs)
        self._axis = axis
        self._squeeze_axis = squeeze_axis

    def forward(self, x):
        return mx.np.split(x, axis=self._axis, num_outputs=4, squeeze_axis=self._squeeze_axis)


@use_np
class BBoxArea(gluon.HybridBlock):
    """Calculate the area of bounding boxes.

    Parameters
    ----------
    fmt : str, default is corner
        Bounding box format, can be {'center', 'corner'}.
        'center': {x, y, width, height}
        'corner': {xmin, ymin, xmax, ymax}
    axis : int, default is -1
        Effective axis of the bounding box. Default is -1(the last dimension).

    Returns
    -------
    A BxNx1 NDArray

    """

    def __init__(self, axis=-1, fmt='corner', **kwargs):
        super(BBoxArea, self).__init__(**kwargs)
        if fmt.lower() == 'corner':
            self._pre = BBoxCornerToCenter(split=True)
        elif fmt.lower() == 'center':
            self._pre = BBoxSplit(axis=axis)
        else:
            raise ValueError("Unsupported format: {}. Use 'corner' or 'center'.".format(fmt))

    def forward(self, x):
        _, _, width, height = self._pre(x)
        width = mx.np.where(width > 0, width, mx.np.zeros_like(width))
        height = mx.np.where(height > 0, height, mx.np.zeros_like(height))
        return width * height


@use_np
class BBoxBatchIOU(gluon.HybridBlock):
    """Batch Bounding Box IOU.

    Parameters
    ----------
    axis : int
        On which axis is the length-4 bounding box dimension.
    fmt : str
        BBox encoding format, can be 'corner' or 'center'.
        'corner': (xmin, ymin, xmax, ymax)
        'center': (center_x, center_y, width, height)
    offset : float, default is 0
        Offset is used if +1 is desired for computing width and height, otherwise use 0.
    eps : float, default is 1e-15
        Very small number to avoid division by 0.

    """

    def __init__(self, axis=-1, fmt='corner', offset=0, eps=1e-15, **kwargs):
        super(BBoxBatchIOU, self).__init__(**kwargs)
        self._offset = offset
        self._eps = eps
        if fmt.lower() == 'center':
            self._pre = BBoxCenterToCorner(split=True)
        elif fmt.lower() == 'corner':
            self._pre = BBoxSplit(axis=axis, squeeze_axis=True)
        else:
            raise ValueError("Unsupported format: {}. Use 'corner' or 'center'.".format(fmt))

    def forward(self, a, b):
        """Compute IOU for each batch

        Parameters
        ----------
        a : mxnet.nd.NDArray or mxnet.sym.Symbol
            (B, N, 4) first input.
        b : mxnet.nd.NDArray or mxnet.sym.Symbol
            (B, M, 4) second input.

        Returns
        -------
        mxnet.nd.NDArray or mxnet.sym.Symbol
            (B, N, M) array of IOUs.

        """
        al, at, ar, ab = self._pre(a)
        bl, bt, br, bb = self._pre(b)

        # (B, N, M)
        left = mx.np.maximum(al.expand_dims(-1), bl.expand_dims(-2))
        right = mx.np.minimum(ar.expand_dims(-1), br.expand_dims(-2))
        top = mx.np.maximum(at.expand_dims(-1), bt.expand_dims(-2))
        bot = mx.np.minimum(ab.expand_dims(-1), bb.expand_dims(-2))

        # clip with (0, float16.max)
        iw = mx.np.clip(right - left + self._offset, a_min=0, a_max=6.55040e+04)
        ih = mx.np.clip(bot - top + self._offset, a_min=0, a_max=6.55040e+04)
        i = iw * ih

        # areas
        area_a = ((ar - al + self._offset) * (ab - at + self._offset)).expand_dims(-1)
        area_b = ((br - bl + self._offset) * (bb - bt + self._offset)).expand_dims(-2)
        union = (area_a + area_b) - i

        return i / (union + self._eps)


@use_np
class BBoxClipToImage(gluon.HybridBlock):
    """Clip bounding box coordinates to image boundaries.
    If multiple images are supplied and padded, must have additional inputs
    of accurate image shape.
    """

    def __init__(self, **kwargs):
        super(BBoxClipToImage, self).__init__(**kwargs)

    def forward(self, x, img):
        """If images are padded, must have additional inputs for clipping

        Parameters
        ----------
        x: (B, N, 4) Bounding box coordinates.
        img: (B, C, H, W) Image tensor.

        Returns
        -------
        (B, N, 4) Bounding box coordinates.

        """
        x = mx.np.maximum(x, 0.0)
        # window [B, 2] -> reverse hw -> tile [B, 4] -> [B, 1, 4], boxes [B, N, 4]
        window = mx.npx.shape_array(img).as_nd_ndarray().slice_axis(axis=0, begin=2, end=None).expand_dims(0).as_np_ndarray()
        m = mx.np.tile(mx.nd.reverse(window.as_nd_ndarray(), axis=1), reps=(2,)).reshape((0, -4, 1, -1)).as_np_ndarray()
        return mx.np.minimum(x, mx.npx.cast(m, dtype='float32'))
