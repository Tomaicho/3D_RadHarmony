"""
MURD - 2.5D Original Implementation in PyTorch
Components file

"""

import torch
import torch.nn as nn


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class Conv2DBlock(nn.Module):
    """2D Convolutional block: Conv -> IN -> LeakyReLU"""
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 use_norm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if use_norm else None
        self.activation = nn.LeakyReLU(0.2, inplace=False)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)
        return x


class ResidualBlock2D(nn.Module):
    """Standard residual block for content encoder"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + residual


class PreActivationResBlock2D(nn.Module):
    """Pre-activation residual block for style encoder (NO instance normalization)"""
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        self.downsample = downsample

        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        
        if downsample:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            )
            self.pool = nn.AvgPool2d(2, 2)
        else:
            if in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.shortcut = nn.Identity()
            self.pool = nn.Identity()
    
    def forward(self, x):
        out = self.main(x)
        if self.downsample:
            out = self.pool(out)
        return out + self.shortcut(x)


# ============================================================================
# ADAPTIVE INSTANCE NORMALIZATION (FOR CONTENT AND STYLE FEATURES CONCATENATION)
# ============================================================================

class AdaptiveInstanceNorm2D(nn.Module):
    """
    Adaptive Instance Normalization
    Modulates content features with style information
    """
    def __init__(self, num_features: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        # Style features are mapped to scale (gamma) and shift (beta)
        self.fc = nn.Linear(style_dim, num_features * 2)
    
    def forward(self, content, style):
        # content: [B, C, H, W]
        # style: [B, style_dim]
        
        # Normalize content
        normalized = self.norm(content)
        
        # Get style parameters
        style_params = self.fc(style)  # [B, 2*C]
        gamma, beta = style_params.chunk(2, dim=1)  # Each [B, C]
        
        # Reshape for broadcasting
        gamma = gamma.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        beta = beta.unsqueeze(2).unsqueeze(3)    # [B, C, 1, 1]
        
        # Apply affine transformation
        return gamma * normalized + beta


class AdaINResidualBlock2D(nn.Module):
    """
    Residual block with AdaIN - Used in Generator
    """
    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.adain1 = AdaptiveInstanceNorm2D(channels, style_dim)
        self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.adain2 = AdaptiveInstanceNorm2D(channels, style_dim)
    
    def forward(self, content, style):
        residual = content
        
        out = self.conv1(content)
        out = self.adain1(out, style)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.adain2(out, style)
        
        return out + residual


# ============================================================================
# CONTENT ENCODER
# ============================================================================

class ContentEncoder2D(nn.Module):
    """
    Content Encoder - Extracts anatomical features
    Uses Instance Normalization
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        
        # Initial conv blocks
        self.conv1 = Conv2DBlock(in_channels, base_channels, 7, 1, 3)
        self.conv2 = Conv2DBlock(base_channels, base_channels * 2, 4, 2, 1)
        self.conv3 = Conv2DBlock(base_channels * 2, base_channels * 4, 4, 2, 1)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock2D(base_channels * 4) for _ in range(4)
        ])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_blocks(x)
        return x


# ============================================================================
# STYLE ENCODER
# ============================================================================

class StyleEncoder2D(nn.Module):
    """
    Style Encoder - Extracts appearance features
    NO Instance Normalization
    """
    def __init__(self, in_channels: int = 3, num_sites: int = 3, 
                 style_dim: int = 64, base_channels: int = 32):
        super().__init__()
        self.num_sites = num_sites
        
        # Initial conv (no normalization!)
        self.initial_conv = nn.Conv2d(in_channels, base_channels, 4, 2, 1)
        
        # Pre-activation residual blocks (no normalization!)
        self.res_block1 = PreActivationResBlock2D(base_channels, base_channels * 2, downsample=True)
        self.res_block2 = PreActivationResBlock2D(base_channels * 2, base_channels * 4, downsample=True)
        self.res_block3 = PreActivationResBlock2D(base_channels * 4, base_channels * 8, downsample=True)
        self.res_block4 = PreActivationResBlock2D(base_channels * 8, base_channels * 16, downsample=True)
        
        # Global average pooling
        self.gap = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Site-specific fully connected layers
        self.site_fcs = nn.ModuleList([
            nn.Linear(base_channels * 16, style_dim) for _ in range(num_sites)
        ])
    
    def forward(self, x, site_idx: int):
        x = self.initial_conv(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        style = self.site_fcs[site_idx](x)
        return style


# ============================================================================
# GENERATOR (WITH AdaIN)
# ============================================================================

class Generator2D(nn.Module):
    """
    Generator - Combines content and style using AdaIN
    """
    def __init__(self, content_channels: int = 256, style_dim: int = 64, out_channels: int = 3):
        super().__init__()
        
        # AdaIN residual blocks (THE KEY COMPONENT!)
        self.res_blocks = nn.ModuleList([
            AdaINResidualBlock2D(content_channels, style_dim) for _ in range(4)
        ])
        
        # Upsampling blocks
        self.deconv1 = nn.ConvTranspose2d(content_channels, content_channels // 2, 4, 2, 1)
        self.norm1 = nn.InstanceNorm2d(content_channels // 2, affine=True)
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        
        self.deconv2 = nn.ConvTranspose2d(content_channels // 2, content_channels // 4, 4, 2, 1)
        self.norm2 = nn.InstanceNorm2d(content_channels // 4, affine=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)
        
        # Final output
        self.final_conv = nn.Conv2d(content_channels // 4, out_channels, 7, 1, 3) # THIS DIFFERS FROM THE PAPER FOR SHAPE MATCHING!!
        # self.final_conv = nn.Conv2d(content_channels // 4, out_channels, 4, 2, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, content, style):
        # Apply AdaIN residual blocks
        x = content
        for res_block in self.res_blocks:
            x = res_block(x, style)
        
        # Upsample
        x = self.deconv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        # Output
        x = self.final_conv(x)
        x = self.tanh(x)
        
        return x


# ============================================================================
# STYLE GENERATOR
# ============================================================================

class StyleGenerator2D(nn.Module):
    """Style Generator - Generates random styles for a site"""
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 256, 
                 style_dim: int = 64, num_sites: int = 3):
        super().__init__()
        self.num_sites = num_sites
        
        # Shared MLP
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=False)
        )
        
        # Site-specific outputs
        self.site_outputs = nn.ModuleList([
            nn.Linear(hidden_dim, style_dim) for _ in range(num_sites)
        ])
    
    def forward(self, z, site_idx: int):
        features = self.shared(z)
        style = self.site_outputs[site_idx](features)
        return style


# ============================================================================
# DISCRIMINATOR
# ============================================================================

class Discriminator2D(nn.Module):
    """Discriminator - Distinguishes real from fake images"""
    def __init__(self, in_channels: int = 3, num_sites: int = 3, base_channels: int = 64):
        super().__init__()
        self.num_sites = num_sites
        
        # Shared convolutional blocks
        self.conv1 = nn.Conv2d(in_channels, base_channels, 4, 2, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)
        self.norm2 = nn.InstanceNorm2d(base_channels * 2, affine=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)
        
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 1, 1)
        self.norm3 = nn.InstanceNorm2d(base_channels * 2, affine=True)
        self.relu3 = nn.LeakyReLU(0.2, inplace=False)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Site-specific classifiers
        self.site_classifiers = nn.ModuleList([
            nn.Linear(base_channels * 2, 1) for _ in range(num_sites)
        ])
    
    def forward(self, x, site_idx: int):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        features = self.relu3(x)
        
        features = self.gap(features)
        features = features.view(features.size(0), -1)
        
        out = self.site_classifiers[site_idx](features)
        return out
