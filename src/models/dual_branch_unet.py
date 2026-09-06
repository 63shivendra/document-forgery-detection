import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvGroupNormAct, ResidualBlock, SRMFeatureExtractor

class DualBranchUNet(nn.Module):
    """
    Dual-Branch U-Net Architecture for Document Forgery & Tampering Detection.
    Branch 1: RGB Spatial Branch (detects visual texture & layout mismatches).
    Branch 2: SRM (Spatial Rich Model) High-Pass Noise Residual Branch (detects camera sensor PRNU noise inconsistencies).
    Uses GroupNormalization and Spatial Dropout to handle internal covariate drift and prevent overfitting.
    """
    def __init__(self, in_channels=3, out_channels=1, base_filters=64, num_groups=32, dropout_prob=0.2):
        super().__init__()
        self.srm_extractor = SRMFeatureExtractor()
        
        # Encoder Level 1
        self.enc1_rgb = ConvGroupNormAct(in_channels, base_filters, num_groups=num_groups)
        self.enc1_freq = ConvGroupNormAct(3, base_filters // 2, num_groups=num_groups)
        self.fusion1 = ConvGroupNormAct(base_filters + base_filters // 2, base_filters, num_groups=num_groups)
        self.res1 = ResidualBlock(base_filters, dropout_prob=dropout_prob, num_groups=num_groups)
        
        # Encoder Level 2
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvGroupNormAct(base_filters, base_filters * 2, num_groups=num_groups)
        self.res2 = ResidualBlock(base_filters * 2, dropout_prob=dropout_prob, num_groups=num_groups)
        
        # Encoder Level 3
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = ConvGroupNormAct(base_filters * 2, base_filters * 4, num_groups=num_groups)
        self.res3 = ResidualBlock(base_filters * 4, dropout_prob=dropout_prob, num_groups=num_groups)
        
        # Bottleneck Level 4
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvGroupNormAct(base_filters * 4, base_filters * 8, num_groups=num_groups)
        self.res_bottleneck = ResidualBlock(base_filters * 8, dropout_prob=dropout_prob, num_groups=num_groups)
        
        # Decoder Level 3
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ConvGroupNormAct(base_filters * 8, base_filters * 4, num_groups=num_groups)
        
        # Decoder Level 2
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvGroupNormAct(base_filters * 4, base_filters * 2, num_groups=num_groups)
        
        # Decoder Level 1
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvGroupNormAct(base_filters * 2, base_filters, num_groups=num_groups)
        
        # Final Output Segmentation Head (1-channel forgery mask)
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. SRM High-Pass Noise Residual Extraction (PRNU Noise Inconsistency)
        freq_feat = self.srm_extractor(x)
        
        # 2. Level 1 Encoding & Branch Fusion
        rgb_x1 = self.enc1_rgb(x)
        freq_x1 = self.enc1_freq(freq_feat)
        x1 = torch.cat([rgb_x1, freq_x1], dim=1)
        x1 = self.fusion1(x1)
        x1 = self.res1(x1)
        
        # 3. Level 2 Encoding
        x2 = self.pool1(x1)
        x2 = self.enc2(x2)
        x2 = self.res2(x2)
        
        # 4. Level 3 Encoding
        x3 = self.pool2(x2)
        x3 = self.enc3(x3)
        x3 = self.res3(x3)
        
        # 5. Bottleneck
        b = self.pool3(x3)
        b = self.bottleneck(b)
        b = self.res_bottleneck(b)
        
        # 6. Decoding with Skip Connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)
        
        # Output Logits
        logits = self.final_conv(d1)
        return logits
