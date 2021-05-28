# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import division

import torch
from memory_profiler import profile

class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

def to_image_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        return tensors
    
    elif isinstance(tensors, torch.Tensor) or isinstance(tensors, torch.FloatTensor):
        # single tensor shape can be inferred
        #I think should always go here unless we start doing inference on multiple images
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        #torch.Size([1, 3, 800, 1088])

        #getting to this code will cause the program to start to run out of memory -> I think its the new() tensor call
        #if a single image instance is run at a time it should never get here as the previous if statement will be called
        # batched_imgs = tensors[0].new(*batch_shape).zero_()
        # batched_imgs = tensors[0].new_full((*batch_shape), 0)
        # print(batched_imgs)
        # print(*batch_shape)
        # for img, pad_img in zip(tensors, batched_imgs):
        #     pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        # image_sizes = [im.shape[-2:] for im in tensors]

        # return ImageList(batched_imgs, image_sizes)
        return None
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))
