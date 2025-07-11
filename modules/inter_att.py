import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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
        return y


class CrossModalAttention(nn.Module):
    """ Modified CMA attention layer, incorporating Sobel operator, independent conv layers, and learnable attention masks """
    def __init__(self, in_dim, activation=None, ratio=8, cross_value=True, height=64, width=64):
        super(CrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.cross_value = cross_value
        self.height = height
        self.width = width

        # Sobel operator
        #self.sobel = CustomSobel()
        
        self.sobel1 = LearnableEEM(in_dim)
        
        self.sobel2 = LearnableEEM(in_dim)

        # First set of conv layers: for cross-attention between x1 and xf2
        self.query_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Second set of conv layers: for cross-attention between x2 and xf1
        self.query_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Learnable attention masks
        self.mask1 = nn.Parameter(torch.zeros(1, height * width, height * width))  # Mask for attention1
        self.mask2 = nn.Parameter(torch.zeros(1, height * width, height * width))  # Mask for attention2

        self.gamma1 = nn.Parameter(torch.ones(1))  # For scaling out1
        self.gamma2 = nn.Parameter(torch.ones(1))  # For scaling out2

        self.softmax = nn.Softmax(dim=-1)

        # Initialize conv layer weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
            

    def forward(self, x1, x2):
        """
        Inputs:
            x1: First input feature map (B, C, H, W)
            x2: Second input feature map (B, C, H, W)
        Outputs:
            out1: Enhanced x1 feature map (based on cross-attention between x1 and xf2)
            out2: Enhanced x2 feature map (based on cross-attention between x2 and xf1)
        """
        B, C, H, W = x1.size()

        # Verify input dimensions match expected mask size
        assert H == self.height and W == self.width, f"Input size ({H}, {W}) does not match expected ({self.height}, {self.width})"

        # Apply Sobel operator to generate xf1 and xf2
        xf1 = self.sobel1(x1)  # B, C, H, W
        
        xf2 = self.sobel2(x2)  # B, C, H, W

        # Compute cross-attention between x1 and xf2 to enhance x1
        proj_query1 = self.query_conv1(x1).view(B, -1, H*W).permute(0, 2, 1)  # B, HW, C/ratio
        proj_key1 = self.key_conv1(xf2).view(B, -1, H*W)  # B, C/ratio, HW
        energy1 = torch.bmm(proj_query1, proj_key1)  # B, HW, HW
        attention1 = self.softmax(energy1)  # B, HW, HW
        # Apply learnable mask to attention1
        #mask1 = torch.sigmoid(self.mask1)  # 1, HW, HW -> [0, 1]
        #attention1 = attention1 * mask1  # Element-wise multiplication, broadcasting over batch
        if self.cross_value:
            proj_value1 = self.value_conv1(xf2).view(B, -1, H*W)  # B, C, HW
        else:
            proj_value1 = self.value_conv1(x1).view(B, -1, H*W)  # B, C, HW
        out1 = torch.bmm(proj_value1, attention1.permute(0, 2, 1))  # B, C, HW
        out1 = out1.view(B, C, H, W)
        out1 = self.gamma1 * out1 + x1  # Residual connection without gamma
        if self.activation is not None:
            out1 = self.activation(out1)

        # Compute cross-attention between x2 and xf1 to enhance x2
        proj_query2 = self.query_conv2(x2).view(B, -1, H*W).permute(0, 2, 1)  # B, HW, C/ratio
        proj_key2 = self.key_conv2(xf1).view(B, -1, H*W)  # B, C/ratio, HW
        energy2 = torch.bmm(proj_query2, proj_key2)  # B, HW, HW
        attention2 = self.softmax(energy2)  # B, HW, HW
        # Apply learnable mask to attention2
        #mask2 = torch.sigmoid(self.mask2)  # 1, HW, HW -> [0, 1]
        #attention2 = attention2 * mask2  # Element-wise multiplication, broadcasting over batch
        if self.cross_value:
            proj_value2 = self.value_conv2(xf1).view(B, -1, H*W)  # B, C, HW
        else:
            proj_value2 = self.value_conv2(x2).view(B, -1, H*W)  # B, C, HW
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))  # B, C, HW
        out2 = out2.view(B, C, H, W)
        out2 = self.gamma2 * out2 + x2  # Residual connection without gamma
        if self.activation is not None:
            out2 = self.activation(out2)

        return out1, out2

if __name__ == '__main__':

    x = torch.rand(10, 768, 16, 16)
    y = torch.rand(10, 768, 16, 16)
    dcma = CrossModalAttention(768, height=16, width=16)
    out_x, out_y= dcma(x, y)
    print(out_y.size())
    print(out_x.size())