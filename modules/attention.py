import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusionModule(nn.Module):
    """
    Attention Fusion Module - Performs multi-directional compression, fusion, and attention processing on two input feature maps
    """

    def __init__(self, channels):
        super(AttentionFusionModule, self).__init__()
        self.channels = channels

        # Define convolutional layers for compressing different dimensions
        # Dimension 1: Channel dimension (1*h*w)
        self.conv_channel = nn.Conv2d(2, 1, kernel_size=1, bias=False)

        # Dimension 2: Height dimension (2c*h*1)
        self.conv_height = nn.Conv2d(2, 1, kernel_size=1, bias=False)

        # Dimension 3: Width dimension (2c*1*w)
        self.conv_width = nn.Conv2d(2, 1, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.bn3 = nn.BatchNorm2d(channels * 2)

        self.fusion = nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels * 2)
        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # Input x1, x2 dimensions: [b, c, h, w]
        b, c, h, w = x1.size()

        # Concatenate the two feature maps along the channel dimension
        x_concat = torch.cat([x1, x2], dim=1)  # [b, 2c, h, w]

        # Dimension 1: Channel dimension compression
        # Global average pooling and max pooling
        avg_pool_channel = torch.mean(x_concat, dim=1, keepdim=True)  # [b, 1, h, w]
        max_pool_channel = torch.max(x_concat, dim=1, keepdim=True)[0]  # [b, 1, h, w]

        # Concatenate the two pooling results
        pool_channel = torch.cat([avg_pool_channel, max_pool_channel], dim=1)  # [b, 2, h, w]

        # Use convolution to compress to 1*h*w
        channel_attention = self.conv_channel(pool_channel)  # [b, 1, h, w]
        channel_attention = self.bn1(channel_attention)
        channel_attention = self.sigmoid(channel_attention)
        #channel_attention = self.relu(channel_attention)

        channel_attention = channel_attention.repeat(1, 2*c, 1, 1)

        # Dimension 2: Height dimension compression
        # Global average pooling and max pooling (along the width dimension)
        avg_pool_height = F.adaptive_avg_pool2d(x_concat, (h, 1))  # [b, c, h, 1]
        max_pool_height = F.adaptive_max_pool2d(x_concat, (h, 1))  # [b, c, h, 1]

        # Concatenate the two pooling results - along the last dimension
        pool_height = torch.cat([avg_pool_height, max_pool_height], dim=3)  # [b, c, h, 2]

        # Rearrange dimensions for convolution - convert channel and height dimensions to batch dimension [b2ch]
        pool_height_reshaped = pool_height.permute(0, 3, 1, 2)

        # Use convolution to compress [b, 1, 2c, h]
        height_attention_reshaped = self.conv_height(pool_height_reshaped)

        # Restore original shape
        height_attention = height_attention_reshaped.permute(0, 2, 3, 1)
        height_attention = self.bn2(height_attention)
        height_attention = self.sigmoid(height_attention)
        #height_attention = self.relu(height_attention)

        height_attention = height_attention.repeat(1, 1, 1, w)

        # Dimension 3: Width dimension compression
        # Global average pooling and max pooling (along the height dimension)
        avg_pool_width = F.adaptive_avg_pool2d(x_concat, (1, w))  # [b, 2c, 1, w]
        max_pool_width = F.adaptive_max_pool2d(x_concat, (1, w))  # [b, 2c, 1, w]

        # Concatenate the two pooling results - along the second-to-last dimension
        pool_width = torch.cat([avg_pool_width, max_pool_width], dim=2)  # [b, 2c, 2, w]

        # Rearrange dimensions for convolution - convert channel and width dimensions to batch dimension
        # [b, 2c, 2, w] -> [b*2c*w, 1, 2, 1]
        pool_width_reshaped = pool_width.permute(0, 2, 1, 3)

        # Use convolution to compress [b, 1, 2c, w]
        width_attention_reshaped = self.conv_width(pool_width_reshaped)

        # Restore original shape [b, 2c, 1, w]
        width_attention = width_attention_reshaped.permute(0, 2, 1, 3)
        width_attention = self.bn3(width_attention)
        width_attention = self.sigmoid(width_attention)
        #width_attention = self.relu(width_attention)

        width_attention = width_attention.repeat(1, 1, h, 1)

        # Multiply the three dimensions to obtain attention weights
        attention_weights = channel_attention * height_attention * width_attention  # [b, 2c, h, w]

        # Normalize using sigmoid
        #attention_weights = torch.sigmoid(attention_weights)  # [b, 2c, h, w]

        # Weight the original feature map
        output = x_concat * attention_weights  # [b, 2c, h, w]
        
        # Fuse the feature map
        output = self.fusion(output)  # [b, c, h, w]
        output = self.bn(output)
        output = self.relu(output)
        
        out1 = output[:, :c, :, :]  # ǰ�벿�� (b, c, h, w)
        out2 = output[:, c:, :, :]  # ��벿�� (b, c, h, w)

        return out1, out2


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
        
        
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, out_dim=None, add=False, ratio=8):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.add = add
        if out_dim is None:
            out_dim = in_dim
        self.out_dim = out_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X C X(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, self.out_dim, width, height)

        if self.add:
            out = self.gamma*out + x
        else:
            out = self.gamma*out
        return out  # , attention

class ChannelwiseCosineSimilarityScoring(nn.Module):
    def __init__(self, in_planes, ratio, temperature=1.0, epsilon=1e-6):
        super(ChannelwiseCosineSimilarityScoring, self).__init__()
        
        self.temperature = temperature
        self.epsilon = epsilon

        #self.channel_attention1 = ChannelAttention(in_planes, ratio)
        #self.channel_attention2 = ChannelAttention(in_planes, ratio)
        
        #self.spatial_attention1 = SpatialAttention()
        #self.spatial_attention2 = SpatialAttention()
        
        
        #self.self_att1 = Self_Attn(in_planes)
        #self.self_att2 = Self_Attn(in_planes)
        
        self.fusion = AttentionFusionModule(channels=in_planes)

    def forward(self, x1, x2):
        batch_size, num_channels, height, width = x1.shape
        
        x1, x2 = self.fusion(x1, x2)

        #channel1 = self.channel_attention1(x1)
        #channel2 = self.channel_attention2(x2)
        
        #spatial1 = self.spatial_attention1(x1)
        #spatial2 = self.spatial_attention2(x2)
        
        #xs1 = self.self_att1(x1)
        
        #xs2 = self.self_att2(x2)

        # Flatten feature maps to [B, C, H*W]
        x1_flat = x1.view(batch_size, num_channels, -1)  # [B, C, H*W]
        x2_flat = x2.view(batch_size, num_channels, -1)  # [B, C, H*W]

        # Batch compute cosine similarity
        # Dot product
        dot_product = torch.sum(x1_flat * x2_flat, dim=-1)  # [B, C]

        # Compute norms
        norm_x1 = torch.norm(x1_flat, dim=-1)  # [B, C]
        norm_x2 = torch.norm(x2_flat, dim=-1)  # [B, C]

        # Cosine similarity
        cos_sims = dot_product / (norm_x1 * norm_x2 + self.epsilon)  # [B, C]

        shifted_sims = cos_sims

        # For each batch, find the maximum and minimum similarity
        min_sims, _ = torch.min(shifted_sims, dim=1, keepdim=True)  # [B, 1]
        max_sims, _ = torch.max(shifted_sims, dim=1, keepdim=True)  # [B, 1]

        # Compute the ratio of similarities (normalize to [0, 1])
        sim_range = max_sims - min_sims + self.epsilon
        normalized_sims = (shifted_sims - min_sims) / sim_range

        # Invert normalized similarities so that lower similarity yields higher scores
        inverted_sims = normalized_sims

        # Use sigmoid to adjust score distribution
        channel_scores = torch.sigmoid(inverted_sims / self.temperature)

        # Reshape scores to [B, C, 1, 1] for multiplication with feature maps
        channel_scores_reshaped = channel_scores.view(batch_size, num_channels, 1, 1)

        xe1 = x1 * channel_scores_reshaped + x1

        xe2 = x2 * channel_scores_reshaped + x2
        
        
        x = xe1 + xe2
        

        return x


# Usage example
def demo():
    # Create two random feature maps
    batch_size, channels, height, width = 32, 768, 7, 7
    feat1 = torch.randn(batch_size, channels, height, width)
    feat2 = torch.randn(batch_size, channels, height, width)

    # Instantiate the module
    similarity_module = ChannelwiseCosineSimilarityScoring(in_planes=768, ratio=8, temperature=0.5)

    # Obtain weighted features and channel scores
    o1, o2 = similarity_module(feat1, feat2)

    print(f"Input feature shape: {feat1.shape}")
    print(f"Channel score shape: {o1.shape}")
    print(f"Weighted feature shape: {o2.shape}")


if __name__ == "__main__":
    demo()