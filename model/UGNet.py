from model.pvtv2 import *
import torch.nn.functional as F

from torch import nn
import torch

from .dysample import DySample
#from .baseline_Feature_Enhance import Decoder_MS_Block

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MS_Blcok(nn.Module):
 

    def __init__(self, channels=64, r=4):
        super(MS_Blcok, self).__init__()
        inter_channels = int(channels // r)
        ##self.conv_cat = nn.Conv2d(channels * 2, channels, 3, 1, 1)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = torch.cat((high_feat, low_feat), dim=1)
        # x = self.conv_cat(x)
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        out = x + x * wei

        return out


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)
    
# class Up(nn.Module):
#     """Upscaling then double conv"""
#     def __init__(self, high_channels, low_channels, out_channels, bilinear=True):
#         super().__init__()
#         in_channels=high_channels+low_channels
#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             #self.up =  nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#             self.up = DySample(high_channels)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


class Map_Generate(nn.Module):
    def __init__(self, mean, std):
        super(Map_Generate, self).__init__()
        self.prob = nn.Sigmoid()
        # 高斯函数变换,根据需要调整标准差
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, map):
        prob_map = self.prob(map)

        fore_uncertainty_map = prob_map - 0.5
        back_uncertainty_map = 0.5 - prob_map

        # Step 3: Apply Gaussian function transformation
        gauss_fore_map = torch.exp(
            -((fore_uncertainty_map - self.mean.to(map.device)) ** 2) / (2 * (self.std.to(map.device) ** 2)))
        gauss_back_map = torch.exp(
            -((back_uncertainty_map - self.mean.to(map.device)) ** 2) / (2 * (self.std.to(map.device) ** 2)))

        # Step 4: Multiply Gaussian maps
        fore_rank_map = gauss_fore_map 
        back_rank_map = gauss_back_map 

        return fore_rank_map.detach(), back_rank_map.detach()


class Uncertainty_Decoder_Block(nn.Module):
    def __init__(self, high_channel, low_channel, out_channel, num_classes):
        super(Uncertainty_Decoder_Block, self).__init__()
        self.rank = Map_Generate(0.7, 0.1)
        self.high_channel = high_channel
        self.low_channel = low_channel
        self.out_channel = out_channel
        self.conv_high = BasicConv2d(2 * self.high_channel, self.high_channel, 3, 1, 1)
        self.conv_low = BasicConv2d(2 * self.low_channel, self.low_channel, 3, 1, 1)
        self.conv_fusion = nn.Conv2d(self.high_channel+self.low_channel, self.out_channel, 3, 1, 1)
        self.ms_block = MS_Blcok(channels=self.high_channel)
        self.seg_out = nn.Conv2d(self.out_channel, num_classes, 1)
        
        self.upsampling = DySample(self.high_channel)

        #self.up_fuse=Up(high_channel,low_channel,out_channel)

    def forward(self, feature_low, feature_high, map):
        # map = map[:, 1, :, :].unsqueeze(1)
        # if edge.dim() != 4:
        #     edge = edge.unsqueeze(1)
        # edge = F.interpolate(edge, feature_high.size()[2:], mode='bilinear', align_corners=True)
        uncertainty_fore_map_high, uncertainty_back_map_high = self.rank(map)

        fore_map_high = 5 * uncertainty_fore_map_high * feature_high
        back_map_high = 5*uncertainty_back_map_high * feature_high
        uncertainty_feature_high = torch.cat((fore_map_high, back_map_high), dim=1)
        uncertainty_feature_high=self.conv_high(uncertainty_feature_high)+feature_high
        uncertainty_feature_high = self.ms_block(uncertainty_feature_high)

        #uncertainty_high_up = F.interpolate(uncertainty_feature_high, feature_low.size()[2:], mode='bilinear',align_corners=True)
        
        uncertainty_high_up = self.upsampling(uncertainty_feature_high)

        low_map = F.interpolate(map, feature_low.size()[2:], mode='bilinear', align_corners=True)
        #edge = F.interpolate(edge, feature_low.size()[2:], mode='bilinear', align_corners=True)
        uncertainty_fore_map_low, uncertainty_back_map_low = self.rank(low_map)
        fore_map_low = 5*uncertainty_fore_map_low * feature_low
        back_map_low = 5*uncertainty_back_map_low * feature_low
        uncertainty_feature_low = torch.cat((fore_map_low, back_map_low), dim=1)
        uncertainty_low = self.conv_low(uncertainty_feature_low)+feature_low

        seg_fusion = torch.cat((uncertainty_high_up, uncertainty_low), dim=1)
        seg_fusion = self.conv_fusion(seg_fusion)
        seg = self.seg_out(seg_fusion)

        return seg_fusion, seg

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Encoder
        backbone = pvt_v2_b2()
        model_dict = backbone.state_dict()
        path = 'pretrain/pvt_v2_b2.pth'
        save_model = torch.load(path, map_location='cpu')
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)  # 64, 128 , 320 ,512
        self.encoder = backbone
        print('Pretrained encoder loaded.')

        # neck模块

        # self.reduce1 = nn.Conv2d(64, 64, 1)
        # self.reduce4 = nn.Conv2d(512, 256, 1)
        # self.conv_edge = nn.Sequential(
        #     BasicConv2d(256 + 64, 256, 3, padding=1),
        #     BasicConv2d(256, 256, 3, padding=1),
        #     nn.Conv2d(256, 1, 1))

        self.conv_map = nn.Sequential(
            BasicConv2d(512, 256, 3, padding=1),
            nn.Conv2d(256, 1, 1)
        )
        # Decoder模块
        self.decoder3 = Uncertainty_Decoder_Block(512, 320, 320, 1)
        self.decoder2 = Uncertainty_Decoder_Block(320, 128, 128, 1)
        self.decoder1 = Uncertainty_Decoder_Block(128, 64, 64, 1)
        #self.decoder2=Decoder_MS_Block(320, 128, 128)
        #self.decoder1=Decoder_MS_Block(128, 64, 64)

        # sal head
        self.sigmoid = nn.Sigmoid()

    def upsample(self, x, shape):
        return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]

        # backbone2
        x4, x3, x2, x1 = self.encoder(x)

        # neck
        # 边缘生成
        # edge1 = self.reduce1(x1)
        # edge4 = self.reduce4(x4)
        # edge4 = F.interpolate(edge4, edge1.size()[2:], mode='bilinear', align_corners=False)
        # edge = torch.cat((edge4, edge1), dim=1)
        # edge = self.conv_edge(edge)
        # edge=self.sigmoid(edge)

        # Decoder
        predict4 = F.interpolate(self.conv_map(x4), x4.size()[2:], mode='bilinear', align_corners=True)
        sal4 = self.upsample(predict4, size)
        sal_sig4 = self.sigmoid(sal4)

        fusion, predict3 = self.decoder3(x3, x4, predict4)
        sal3 = self.upsample(predict3, size)
        sal_sig3 = self.sigmoid(sal3)

        fusion, predict2 = self.decoder2(x2, fusion, predict3)
        #fusion, predict2 = self.decoder2(x2, fusion)
        sal2 = self.upsample(predict2, size)
        sal_sig2 = self.sigmoid(sal2)

        fusion, predict1 = self.decoder1(x1, fusion, predict2)
        #fusion, predict1 = self.decoder1(x1, fusion)
        sal1 = self.upsample(predict1, size)
        sal_sig1 = self.sigmoid(sal1)

        # return sal4, sal_sig4, sal3, sal_sig3, sal2, sal_sig2, sal1, sal_sig1, self.upsample(edge, size)
        return sal4, sal_sig4, sal3, sal_sig3, sal2, sal_sig2, sal1, sal_sig1
