import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseAttenGraspNet(nn.Module):

    def __init__(self, input_channels=1, dropout=False, prob=0.0, bottleneck=True):
        super(DenseAttenGraspNet, self).__init__()

        if bottleneck == True:
            block = BottleneckBlock
        else:
            block = BasicBlock

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=3, padding=3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        nb_layers = 16  # 16     layers can be much more   now the best is 16
        input_channels = 128
        growth_rate = 24  # 12  now the best is 24
        self.block = DenseBlock(block, nb_layers=nb_layers, input_channels=input_channels, growth_rate=growth_rate, dropRate=prob)

        self.change_channel = nn.Conv2d(input_channels + nb_layers * growth_rate, 128, kernel_size=1)

        self.channel_attention1 = ChannelAttention(in_planes=128)
        self.spatial_attention1 = SpatialAttention()

        # self.gam_attention1 = GAM_Attention(128, 128)

        self.attention1 = Self_Attn(128)

        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.attention2 = Self_Attn(64)

        self.channel_attention2 = ChannelAttention(in_planes=64)
        self.spatial_attention2 = SpatialAttention()

        # self.gam_attention2 = GAM_Attention(64, 64)


        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.attention3 = Self_Attn(32)

        self.channel_attention3 = ChannelAttention(in_planes=32)
        self.spatial_attention3 = SpatialAttention()

        # self.gam_attention3 = GAM_Attention(32, 32)


        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=9, stride=3, padding=3, output_padding=1)

        self.pos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.cos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.sin_output = nn.Conv2d(32, 1, kernel_size=2)
        self.width_output = nn.Conv2d(32, 1, kernel_size=2)

        self.dropout1 = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        query_x3 = x
        #100 * 100
        x = F.relu(self.bn2(self.conv2(x)))
        query_x2 = x
        #50 * 50
        x = F.relu(self.bn3(self.conv3(x)))
        query_x1 = x
        #25 * 25
        x = F.relu(self.block(x))
        x = F.relu(self.change_channel(x))
        # attention_x1 = self.attention1(x)
        channel_x = F.relu(self.channel_attention1(x))
        spatial_x = F.relu(self.spatial_attention1(query_x1))
        x = torch.add(channel_x, spatial_x)
        # x = F.relu(self.gam_attention1(x))
        #25 * 25
        # x = F.relu(self.bn4(self.conv4(attention_x1)))
        x = F.relu(self.bn4(self.conv4(x)))
        channel_x = F.relu(self.channel_attention2(x))
        spatial_x = F.relu(self.spatial_attention2(query_x2))
        x = torch.add(channel_x, spatial_x)
        # x = F.relu(self.gam_attention2(x))

        # attention_x2 = F.relu(self.attention2(x))
        # attention_x2 = self.attention2(x)
        #50 * 50
        x = F.relu(self.bn5(self.conv5(x)))
        # attention_x3 = F.relu(self.attention3(x, query_x3))
        # attention_x3 = self.attention3(x)
        channel_x = F.relu(self.channel_attention3(x))
        spatial_x = F.relu(self.spatial_attention3(query_x3))
        x = torch.add(channel_x, spatial_x)
        # x = F.relu(self.gam_attention3(x))

        #100 * 100
        x = self.conv6(x)

        pos_output = self.pos_output(self.dropout1(x))
        cos_output = self.cos_output(self.dropout1(x))
        sin_output = self.sin_output(self.dropout1(x))
        width_output = self.width_output(self.dropout1(x))

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)
        # p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        # cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        # sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        # width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

class BasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = output_channels * 4
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, output_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, block, nb_layers=8, input_channels=128, growth_rate=16, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, input_channels, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, input_channels, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(input_channels+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x_input):
        """
            inputs :
                x_input : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x_input.size()
        proj_query = self.query_conv(x_input).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x_input).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        # attention = self.sigmoid(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x_input).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # out = out + x_input
        # out = self.gamma * out + x_input
        out = self.gamma * out
        # return out, attention
        return out



class Attention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, key_in_dim, query_in_dim):
        super(Attention, self).__init__()
        self.key_channel_in = key_in_dim
        self.query_channel_in = query_in_dim

        self.query_conv = nn.Conv2d(in_channels=query_in_dim, out_channels=query_in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=key_in_dim, out_channels=key_in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=key_in_dim, out_channels=key_in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x_input, x_query):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x_input.size()
        proj_query = self.query_conv(x_query).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x_input).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x_input).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x_input
        # return out, attention
        return out

class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        temp_x = torch.cat([avg_out, max_out], dim=1)
        temp_x = self.conv1(temp_x)
        attention = self.sigmoid(temp_x)
        x = attention * x
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        out = attention * x
        return out

class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out