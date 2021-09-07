import random
import glob
import os
import PIL.ImageFilter
import cv2
import inspect
import numpy as np
from PIL import Image, ImageFilter
import augly.image.functional as F
import augly.utils as utils

import torch
from torchvision import transforms as _transforms

from openselfsup.utils import build_from_cfg

from ..registry import PIPELINES

# register all existing transforms in torchvision
_EXCLUDED_TRANSFORMS = ['GaussianBlur']
for m in inspect.getmembers(_transforms, inspect.isclass):
    if m[0] not in _EXCLUDED_TRANSFORMS:
        PIPELINES.register_module(m[1])


@PIPELINES.register_module
class RandomAppliedTrans(object):
    """Randomly applied transformations.

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


# custom transforms
@PIPELINES.register_module
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)."""

    _IMAGENET_PCA = {
        'eigval':
            torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec':
            torch.Tensor([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])
    }

    def __init__(self):
        self.alphastd = 0.1
        self.eigval = self._IMAGENET_PCA['eigval']
        self.eigvec = self._IMAGENET_PCA['eigvec']

    def __call__(self, img):
        assert isinstance(img, torch.Tensor), \
            "Expect torch.Tensor, got {}".format(type(img))
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

'''
@PIPELINES.register_module
class Resize(object):

    def __init__(self, size):
        self.size=size
        
    def __call__(self, img):
        img = img.resize(self.size)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
'''

@PIPELINES.register_module
class AugLy(object):

    def __init__(self):
        pass

    def _overlay_text(self, img):
        text_length = random.randint(1, 10)
        text = random.sample(list(range(0, 255)), text_length)

        opacity = random.uniform(0.1, 0.9)
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        font_size = random.uniform(0.1, 1.0)

        x_pos = random.uniform(0, 1.0)
        y_pos = random.uniform(0, 1.0)
        img = F.overlay_text(
            img,
            text=text,
            font_file=utils.MEME_DEFAULT_FONT,
            font_size=font_size,
            opacity=opacity,
            color=color,
            x_pos=x_pos,
            y_pos=y_pos)

        img = img.convert('RGB')
        return img

    def _apply_filter(self, img):
        filter_type = random.choice(
            [PIL.ImageFilter.BLUR, PIL.ImageFilter.CONTOUR, PIL.ImageFilter.DETAIL, PIL.ImageFilter.EDGE_ENHANCE,
             PIL.ImageFilter.EDGE_ENHANCE_MORE, PIL.ImageFilter.EMBOSS, PIL.ImageFilter.FIND_EDGES])
        return img.filter(filter_type)

    def _brightness(self, img):
        return F.brightness(img, factor=random.uniform(0.5, 2))

    def _change_contrast(self, img):
        return F.contrast(img, factor=random.uniform(0.5, 2))

    def __call__(self, img):
        fn = random.choice(
            [self._overlay_text, self._apply_filter, self._brightness, self._change_contrast])
        img_output = fn(img)

        return img_output

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class AugLyOverlay(object):

    def __init__(self, src_jpg_path):
        self.jpgs = list(glob.glob(os.path.join(src_jpg_path, '*.jpg')))

    def __call__(self, img):
        if random.uniform(0,1.0)>0.5:
            img_output = self._overlay_image(img)
        else:
            img_output = self._overlay_background(img)
        img_output = img_output.convert('RGB')
        img_output = img_output.resize(img.size)
        return img_output

    def _overlay_image(self, img):

        overlay = random.choice(self.jpgs)
        overlay = PIL.Image.open(overlay)
        img_output = F.overlay_image(img, overlay,
                                     opacity=random.uniform(0.5,1.0),
                                     overlay_size=random.uniform(0.5,1.0),
                                     x_pos=random.uniform(0.1,0.9),
                                     y_pos=random.uniform(0.1,0.9))

        return img_output

    def _overlay_background(self, img):
        background_image = random.choice(self.jpgs)
        background_image = PIL.Image.open(background_image)

        img_output=F.overlay_onto_background_image(
            img,
            background_image,
            opacity=random.uniform(0.5, 1.0),
            overlay_size=random.uniform(0.5, 1.0),
            x_pos=random.uniform(0.1, 0.9),
            y_pos=random.uniform(0.1, 0.9),
            scale_bg=True)

        return img_output


    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class Solarization(object):
    """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733."""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 - img)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
