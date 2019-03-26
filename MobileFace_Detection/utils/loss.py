# pylint: disable=arguments-differ
"""Custom losses.
Losses are subclasses of gluon.loss.Loss which is a HybridBlock actually.
"""
from __future__ import absolute_import
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like


class YOLOV3Loss(Loss):
    """Losses of YOLO v3.

    Parameters
    ----------
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.

    """
    def __init__(self, batch_axis=0, weight=None, **kwargs):
        super(YOLOV3Loss, self).__init__(weight, batch_axis, **kwargs)
        self._sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self._l1_loss = gluon.loss.L1Loss()

    def hybrid_forward(self, F, objness, box_centers, box_scales, cls_preds,
                       objness_t, center_t, scale_t, weight_t, class_t, class_mask):
        """Compute YOLOv3 losses.

        Parameters
        ----------
        objness : mxnet.nd.NDArray
            Predicted objectness (B, N), range (0, 1).
        box_centers : mxnet.nd.NDArray
            Predicted box centers (x, y) (B, N, 2), range (0, 1).
        box_scales : mxnet.nd.NDArray
            Predicted box scales (width, height) (B, N, 2).
        cls_preds : mxnet.nd.NDArray
            Predicted class predictions (B, N, num_class), range (0, 1).
        objness_t : mxnet.nd.NDArray
            Objectness target, (B, N), 0 for negative 1 for positive, -1 for ignore.
        center_t : mxnet.nd.NDArray
            Center (x, y) targets (B, N, 2).
        scale_t : mxnet.nd.NDArray
            Scale (width, height) targets (B, N, 2).
        weight_t : mxnet.nd.NDArray
            Loss Multipliers for center and scale targets (B, N, 2).
        class_t : mxnet.nd.NDArray
            Class targets (B, N, num_class).
            It's relaxed one-hot vector, i.e., (1, 0, 1, 0, 0).
            It can contain more than one positive class.
        class_mask : mxnet.nd.NDArray
            0 or 1 mask array to mask out ignored samples (B, N, num_class).

        Returns
        -------
        tuple of NDArrays
            obj_loss: sum of objectness logistic loss
            center_loss: sum of box center logistic regression loss
            scale_loss: sum of box scale l1 loss
            cls_loss: sum of per class logistic loss

        """
        # compute some normalization count, except batch-size
        denorm = F.cast(
            F.shape_array(objness_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        weight_t = F.broadcast_mul(weight_t, objness_t)
        hard_objness_t = F.where(objness_t > 0, F.ones_like(objness_t), objness_t)
        new_objness_mask = F.where(objness_t > 0, objness_t, objness_t >= 0)
        obj_loss = F.broadcast_mul(
            self._sigmoid_ce(objness, hard_objness_t, new_objness_mask), denorm)
        center_loss = F.broadcast_mul(self._sigmoid_ce(box_centers, center_t, weight_t), denorm * 2)
        scale_loss = F.broadcast_mul(self._l1_loss(box_scales, scale_t, weight_t), denorm * 2)
        denorm_class = F.cast(
            F.shape_array(class_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        class_mask = F.broadcast_mul(class_mask, objness_t)
        cls_loss = F.broadcast_mul(self._sigmoid_ce(cls_preds, class_t, class_mask), denorm_class)
        return obj_loss, center_loss, scale_loss, cls_loss
