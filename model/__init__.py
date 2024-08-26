from __future__ import annotations
import torch
from torch import Tensor
from torch import nn
from typing import Any
from encoder import ENCODERS
from decoder import Decoder

class MSVMUNet(nn.Module):
    def __init__(
        self,
        enc_name: str = "tiny_0230s",
        in_channels: int = 3,
        num_classes: int = 9,
        deep_supervision: bool = False,
    ) -> None:
        super(MSVMUNet, self).__init__()
        self.encoder = ENCODERS[enc_name](in_channels=in_channels)
        self.dims = self.encoder.dims
        self.decoder = Decoder(dims=self.dims, num_classes=num_classes, deep_supervision=deep_supervision)
        self.deep_supervision_scales = getattr(self.decoder, "decoder.deep_supervision_scales", None)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        for name, param in self.encoder.named_parameters():
            param.requires_grad = True

def build_model(**kwargs: Any) -> MSVMUNet:
    return MSVMUNet(**kwargs)
