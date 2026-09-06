import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGroupNormAct(nn.Module):
    """
    Convolution + Group Normalization + Activation block.
    Uses GroupNorm (num_groups=32) instead of BatchNorm to prevent internal covariate shift
    and maintain training stability across small batch sizes on GPUs with limited VRAM.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_groups=32):
        super().__init__()
        # Adjust groups if out_channels is smaller than num_groups
        groups = num_groups if out_channels % num_groups == 0 else min(8, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.act = nn.SiLU(inplace=True)  # Swish activation for smooth gradient flow

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual Block with Group Normalization and Spatial Dropout2D.
    Spatial Dropout prevents co-adaptation of adjacent feature map channels (anti-overfitting).
    """
    def __init__(self, channels, dropout_prob=0.2, num_groups=32):
        super().__init__()
        self.conv1 = ConvGroupNormAct(channels, channels, num_groups=num_groups)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        groups = num_groups if channels % num_groups == 0 else min(8, channels)
        self.gn2 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        out += residual
        return self.act(out)


class SRMFeatureExtractor(nn.Module):
    """
    Spatial Rich Model (SRM) High-Pass Noise Filter Bank.
    Extracts sensor noise residuals (PRNU/Noiseprint approximation) using 3 SRM high-pass kernels.
    Sidesteps the low-data problem on small forgery datasets by directly exposing camera noise inconsistencies.
    """
    def __init__(self):
        super().__init__()
        # SRM Kernel 1: 3x3 1st Order Edge Filter
        srm1 = torch.tensor([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ], dtype=torch.float32) / 4.0

        # SRM Kernel 2: 3x3 2nd Order Point Filter
        srm2 = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32) / 8.0

        # SRM Kernel 3: 5x5 3rd Order Edge Filter
        srm3 = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=torch.float32) / 12.0

        # Pad 3x3 kernels to 5x5 for unified Conv2d execution
        srm1_padded = F.pad(srm1, (1, 1, 1, 1)).view(1, 1, 5, 5)
        srm2_padded = F.pad(srm2, (1, 1, 1, 1)).view(1, 1, 5, 5)
        srm3_padded = srm3.view(1, 1, 5, 5)

        srm_bank = torch.cat([srm1_padded, srm2_padded, srm3_padded], dim=0) # [3, 1, 5, 5]
        self.register_buffer('srm_bank', srm_bank)

    def forward(self, x):
        # Convert RGB to Grayscale
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        # Apply SRM Filter Bank to produce 3-channel Noise Residual Map
        noise_residual = F.conv2d(gray, self.srm_bank, padding=2)
        return noise_residual
