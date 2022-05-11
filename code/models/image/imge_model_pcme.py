from third_party.pcme.models.image_encoder import EncoderImage
from third_party.pcme.config import parse_config
import torch.nn as nn


class ImageModelPCME(nn.Module):

    def __init__(self, config):
        super(ImageModelPCME, self).__init__()

        self.config = config.model
        self.encoder = EncoderImage(self.config)

    def forward(self, data, n_samples=1):
        return self.encoder.forward(data, n_samples)
