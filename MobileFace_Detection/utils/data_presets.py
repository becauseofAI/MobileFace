from __future__ import absolute_import
import numpy as np
import mxnet as mx
import image as timage

def transform_test(imgs, short=416, max_size=1024, stride=1, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    short : int, default=416
        Resize image short side to this `short` and keep aspect ratio. Note that yolo network
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our YOLO implementation.
    stride : int, optional, default is 1
        The stride constraint due to precise alignment of bounding box prediction module.
        Image's width and height must be multiples of `stride`. Use `stride = 1` to
        relax this constraint.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        img = timage.resize_short_within(img, short, max_size, mult_base=stride)
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs

def data_trans(img, short=416, max_size=1024, stride=1, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or list of filenames.

    Parameters
    ----------
    img : NDArray
        The NDArray data format for MXNet to process, such as (H, W, C).
    short : int, default=416
        Resize image short side to this `short` and keep aspect ratio. Note that yolo network
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our YOLO implementation.
    stride : int, optional, default is 1
        The stride constraint due to precise alignment of bounding box prediction module.
        Image's width and height must be multiples of `stride`. Use `stride = 1` to
        relax this constraint.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    imgs = [img]
    return transform_test(imgs, short, max_size, stride, mean, std)