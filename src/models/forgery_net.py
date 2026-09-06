import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .blocks import SRMFeatureExtractor


class ForgeryNet(nn.Module):
    """
    ForgeryNet: Production-grade Image Forgery Localization Model.

    Architecture:
    - Encoder  : EfficientNet-B2 pretrained on ImageNet (transfer learning)
    - Decoder  : U-Net decoder with deep skip connections
    - SRM Head : Fixed high-pass noise filter fused at input level
    - Output   : 1-channel logit map (binary forgery mask)

    Why EfficientNet-B2:
    - Pretrained on 1.28M ImageNet images -> strong low-level feature representations
    - Compound scaling (width + depth + resolution) -> efficient for 6GB VRAM
    - Skip connections preserve spatial detail for precise pixel-level localization

    SRM Fusion:
    - SRM extracts camera noise residuals (PRNU) -> exposes copy-paste boundaries
    - Fused as extra channels into the encoder input (3 RGB + 3 SRM = 6ch -> projected back to 3ch)
    - Keeps EfficientNet pretrained weights intact by projecting before feeding
    """

    def __init__(self, out_channels=1, encoder_name='efficientnet-b2', dropout_prob=0.2):
        super().__init__()

        # SRM noise feature extractor (fixed weights, not trained)
        self.srm = SRMFeatureExtractor()

        # Input projection: fuse RGB (3ch) + SRM noise (3ch) -> 3ch for EfficientNet
        self.input_proj = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True)
        )

        # EfficientNet-B2 encoder + U-Net decoder (pretrained ImageNet weights)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=out_channels,
            activation=None,           # Return raw logits
            decoder_use_batchnorm=True,
            decoder_attention_type='scse',  # Squeeze-and-excitation attention in decoder
        )

        # Dropout before final prediction
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        # 1. Extract SRM noise residual
        srm_feat = self.srm(x)                      # [B, 3, H, W]

        # 2. Fuse RGB + SRM -> project to 3ch (preserve EfficientNet input stats)
        fused = torch.cat([x, srm_feat], dim=1)     # [B, 6, H, W]
        fused = self.input_proj(fused)               # [B, 3, H, W]

        # 3. U-Net forward (EfficientNet encoder + decoder + skip connections)
        fused = self.dropout(fused)
        logits = self.unet(fused)                    # [B, 1, H, W]

        return logits
