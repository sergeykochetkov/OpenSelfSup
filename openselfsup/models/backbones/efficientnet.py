"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import torch
from torch import nn
from torch.nn import functional as F
from pytorch2fineai.pytorch_ext.models.efficientnet.model import EfficientNet
from pytorch2fineai.pytorch_ext.models.efficientnet.utils import get_model_params, BlockDecoder

from ..builder import BACKBONES


@BACKBONES.register_module()
class EfficientNetAbbyy(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        
        
        import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args_list=None, blocks_args=None, out_indices=None, global_params=None):
        super().__init__()

        blocks_args, global_params = get_model_params(model_name=blocks_args)
        blocks_args = BlockDecoder.decode(blocks_args_list)

        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._model = EfficientNet(blocks_args, global_params)

        del self._model._conv_head
        del self._model._bn1
        del self._model._fc

        self._out_indices=out_indices

    def init_weights(self, pretrained=False):
        assert not pretrained, 'Not Implemented'

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._model._conv_stem(inputs)
        x = self._model._bn0(x)
        x = self._model._swish(x)

        outs = []
        for idx, block in enumerate(self._model._blocks):
            x = block(x)
            #print(f'{idx} {x.shape}')
            if idx in self._out_indices:
                outs.append(x)
            if idx > max(self._out_indices):
                break

        return [outs[-1]]

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.extract_features(inputs)

        return x
