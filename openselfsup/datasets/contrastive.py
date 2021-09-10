import torch
from PIL import Image
from .registry import DATASETS
from .base import BaseDataset
from .utils import to_numpy
from .registry import PIPELINES
from openselfsup.utils import build_from_cfg
from torchvision.transforms import Compose


@DATASETS.register_module
class ContrastiveDataset(BaseDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline, normalization_pipeline, prefetch=False):
        data_source['return_label'] = False
        super(ContrastiveDataset, self).__init__(data_source, pipeline, prefetch)
        normalization_pipeline = [build_from_cfg(p, PIPELINES) for p in normalization_pipeline]
        self.normalization_pipeline = Compose(normalization_pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
                type(img))
        img1 = self.pipeline(img)
        img1 = self.normalization_pipeline(img1)
        img2 = self.normalization_pipeline(img)
        if self.prefetch:
            img1 = torch.from_numpy(to_numpy(img1))
            img2 = torch.from_numpy(to_numpy(img2))
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
