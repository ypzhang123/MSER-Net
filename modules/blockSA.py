import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


def get_learnable_sobel(in_chan, out_chan):
    # Initialize Sobel filter as the starting point
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    # Convert to learnable parameters
    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)

    # Create learnable convolution layer
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = nn.Parameter(filter_x)  # Set as learnable parameter, default requires_grad=True

    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = nn.Parameter(filter_y)  # Set as learnable parameter, default requires_grad=True

    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))

    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return g


class LearnableEEM(nn.Module):
    def __init__(self, in_channels):
        super(LearnableEEM, self).__init__()
        # Use learnable Sobel filter
        self.sobel_x1, self.sobel_y1 = get_learnable_sobel(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(in_channels)

        # Add convolution layer to further enhance features
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = run_sobel(self.sobel_x1, self.sobel_y1, x)
        y = self.bn(y)
        y = self.relu(y)
        return y


class ScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super(ScaleAttention, self).__init__()
        # Adjust to the correct weight size
        self.scale_weight = nn.Parameter(torch.ones(3))  # Only need 3 numbers to represent weights for three scales
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x1, x2, x3):
        # Stack features from three scales [3, B, C, H, W]
        stacked = torch.stack([x1, x2, x3], dim=0)

        # Calculate weights [3] -> [3, 1, 1, 1, 1] for broadcasting
        weights = self.softmax(self.scale_weight).reshape(3, 1, 1, 1, 1)

        # Apply weights and sum -> [B, C, H, W]
        weighted_sum = torch.sum(stacked * weights, dim=0)
        return weighted_sum


class MultiScaleEEM(nn.Module):
    def __init__(self, in_channels, use_scale_attention=True):
        super(MultiScaleEEM, self).__init__()
        # Edge Enhancement Modules for different scales
        self.eem_scale1 = LearnableEEM(in_channels)  # Original scale
        self.eem_scale2 = LearnableEEM(in_channels)  # 1/2 scale
        self.eem_scale3 = LearnableEEM(in_channels)  # 1/4 scale

        # Add channel attention mechanism for each scale
        self.attention1 = ChannelAttention(in_channels)
        self.attention2 = ChannelAttention(in_channels)
        self.attention3 = ChannelAttention(in_channels)

        # Used to integrate features from different scales
        self.use_scale_attention = use_scale_attention
        if use_scale_attention:
            self.scale_attention = ScaleAttention(in_channels)
            # Additional fusion layer
            self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        else:
            # Traditional fusion method
            self.fusion = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False)

        # Dynamic pooling layer, adaptive to input size
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Original scale processing
        feat_scale1 = self.eem_scale1(x)
        feat_scale1 = feat_scale1 * self.attention1(feat_scale1)

        # 1/2 scale processing
        x_scale2 = self.pool(x)
        feat_scale2 = self.eem_scale2(x_scale2)
        feat_scale2_up = F.interpolate(feat_scale2, size=x.shape[2:], mode='bilinear', align_corners=False)
        feat_scale2_up = feat_scale2_up * self.attention2(feat_scale2_up)

        # 1/4 scale processing
        x_scale3 = self.pool(x_scale2)  # Cascaded pooling, more flexible
        feat_scale3 = self.eem_scale3(x_scale3)
        feat_scale3_up = F.interpolate(feat_scale3, size=x.shape[2:], mode='bilinear', align_corners=False)
        feat_scale3_up = feat_scale3_up * self.attention3(feat_scale3_up)

        if self.use_scale_attention:
            # Use scale attention mechanism to fuse features
            fused_feat = self.scale_attention(feat_scale1, feat_scale2_up, feat_scale3_up)
            output = self.fusion(fused_feat)
        else:
            # Traditional feature concatenation method
            concat_feat = torch.cat([feat_scale1, feat_scale2_up, feat_scale3_up], dim=1)
            output = self.fusion(concat_feat)

        # Final processing
        output = self.bn(output)
        output = self.relu(output)

        # Residual connection
        output = output + x

        return output


if __name__ == '__main__':
    # Test with input size of 32*192*28*28
    x = torch.randn(32, 192, 28, 28)
    mseem = MultiScaleEEM(192, use_scale_attention=True)

    # Check the model
    y = mseem(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")