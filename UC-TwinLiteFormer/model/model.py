import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
import time
from model import config as cfg 
from .swin_adapter import create_swin_adapter
import matplotlib.pyplot as plt
from torch.nn import Module, Conv2d, Parameter, Softmax
import cv2
import os




def patch_split(input, bin_size):
    """
    b c (bh rh) (bw rw) -> b (bh bw) rh rw c
    """
    B, C, H, W = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    rH = H // bin_num_h
    rW = W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0,2,4,3,5,1).contiguous() # [B, bin_num_h, bin_num_w, rH, rW, C]
    out = out.view(B,-1,rH,rW,C) # [B, bin_num_h * bin_num_w, rH, rW, C]
    return out

def patch_recover(input, bin_size):
    """
    b (bh bw) rh rw c -> b c (bh rh) (bw rw)
    """
    B, N, rH, rW, C = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0,5,1,3,2,4).contiguous() # [B, C, bin_num_h, rH, bin_num_w, rW]
    out = out.view(B, C, H, W) # [B, C, H, W]
    return out

class GCN(nn.Module):
    def __init__(self, num_node, num_channel):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.PReLU(num_node)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)
    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out
class SkipTransformerFusion(nn.Module):
    """
    跳跃特征与Transformer特征融合模块
    """
    def __init__(self, skip_channels, transformer_channels, out_channels, norm_layer):
        super(SkipTransformerFusion, self).__init__()
        
        # 跳跃特征处理分支
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.PReLU(out_channels)
        )
        
        # Transformer特征处理分支
        self.transformer_conv = nn.Sequential(
            nn.Conv2d(transformer_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.PReLU(out_channels)
        )
        
        # 注意力机制融合
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 最终融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.PReLU(out_channels)
        )
        
    def forward(self, skip_feat, transformer_feat):
        # 获取Transformer特征的空间尺寸
        _, _, target_h, target_w = transformer_feat.shape
        
        # 调整跳跃特征尺寸以匹配Transformer特征
        if skip_feat.shape[2:] != (target_h, target_w):
            skip_feat_resized = F.interpolate(skip_feat, size=(target_h, target_w), 
                                            mode='bilinear', align_corners=False)
        else:
            skip_feat_resized = skip_feat
        
        # 处理跳跃特征
        skip_processed = self.skip_conv(skip_feat_resized)
        
        # 处理Transformer特征
        transformer_processed = self.transformer_conv(transformer_feat)
        
        # 计算注意力权重
        concat_feat = torch.cat([skip_processed, transformer_processed], dim=1)
        attention_weight = self.attention(concat_feat)
        
        # 加权融合
        weighted_skip = skip_processed * attention_weight
        weighted_transformer = transformer_processed * (1 - attention_weight)
        
        # 最终融合
        fused_feat = torch.cat([weighted_skip, weighted_transformer], dim=1)
        output = self.fusion_conv(fused_feat)
        
        return output

class CAAM(nn.Module):
    """
    Class Activation Attention Module
    """
    def __init__(self, feat_in, num_classes, bin_size, norm_layer):
        super(CAAM, self).__init__()
        feat_inner = feat_in // 2
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        self.dropout = nn.Dropout2d(0.1)
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bin_num = bin_size[0] * bin_size[1]
        self.gcn = GCN(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)
        self.proj_query = nn.Linear(feat_in, feat_inner)
        self.proj_key = nn.Linear(feat_in, feat_inner)
        self.proj_value = nn.Linear(feat_in, feat_inner)
              
        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_inner, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.PReLU(feat_in)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.PReLU(1)

    def forward(self, x):
        cam = self.conv_cam(x) # [B, K, H, W]
        cls_score = self.sigmoid(self.pool_cam(cam)) # [B, K, bin_num_h, bin_num_w]

        residual = x # [B, C, H, W]
        cam = patch_split(cam, self.bin_size) # [B, bin_num_h * bin_num_w, rH, rW, K]
        x = patch_split(x, self.bin_size) # [B, bin_num_h * bin_num_w, rH, rW, C]

        B = cam.shape[0]
        rH = cam.shape[2]
        rW = cam.shape[3]
        K = cam.shape[-1]
        C = x.shape[-1]
        cam = cam.view(B, -1, rH*rW, K) # [B, bin_num_h * bin_num_w, rH * rW, K]
        x = x.view(B, -1, rH*rW, C) # [B, bin_num_h * bin_num_w, rH * rW, C]

        bin_confidence = cls_score.view(B,K,-1).transpose(1,2).unsqueeze(3) # [B, bin_num_h * bin_num_w, K, 1]
        pixel_confidence = F.softmax(cam, dim=2)

        local_feats = torch.matmul(pixel_confidence.transpose(2, 3), x) * bin_confidence # [B, bin_num_h * bin_num_w, K, C]
        local_feats = self.gcn(local_feats) # [B, bin_num_h * bin_num_w, K, C]
        global_feats = self.fuse(local_feats) # [B, 1, K, C]
        global_feats = self.relu(global_feats).repeat(1, x.shape[1], 1, 1) # [B, bin_num_h * bin_num_w, K, C]
        
        query = self.proj_query(x) # [B, bin_num_h * bin_num_w, rH * rW, C//2]
        key = self.proj_key(local_feats) # [B, bin_num_h * bin_num_w, K, C//2]
        value = self.proj_value(global_feats) # [B, bin_num_h * bin_num_w, K, C//2]
        
        aff_map = torch.matmul(query, key.transpose(2, 3)) # [B, bin_num_h * bin_num_w, rH * rW, K]
        aff_map = F.softmax(aff_map, dim=-1)
        out = torch.matmul(aff_map, value) # [B, bin_num_h * bin_num_w, rH * rW, C]
        
        out = out.view(B, -1, rH, rW, value.shape[-1]) # [B, bin_num_h * bin_num_w, rH, rW, C]
        out = patch_recover(out, self.bin_size) # [B, C, H, W]
        
        out_conv = self.conv_out(out)
        out = residual + out_conv
        
        return out



class ConvBatchnormRelu(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize=3, stride=1, groups=1, dropout_rate=0.0, activation='prelu'):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        if activation == 'prelu':
            self.act = nn.PReLU(nOut)
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1)
        elif activation == 'mish':
            self.act = Mish()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        if self.dropout:
            output = self.dropout(output)
        return output


# class C(nn.Module):
#     '''
#     This class is for a convolutional layer.
#     '''

#     def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
#         '''

#         :param nIn: number of input channels
#         :param nOut: number of output channels
#         :param kSize: kernel size
#         :param stride: optional stride rate for down-sampling
#         '''
#         super().__init__()
#         padding = int((kSize - 1) / 2)
#         self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
#                               groups=groups)

#     def forward(self, input):
#         '''
#         :param input: input feature map
#         :return: transformed feature map
#         '''
#         output = self.conv(input)
#         return output

class DilatedConv(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class BatchnormRelu(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.nOut=nOut
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.depthwise = nn.Conv2d(nin, nin, kernel_size, stride, padding, dilation, groups=nin, bias=False)
        self.pointwise = nn.Conv2d(nin, nout, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class StrideESP(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = DilatedConv(nIn, n, 3, 2)
        self.d1 = DilatedConv(n, n1, 3, 1, 1)
        self.d2 = DilatedConv(n, n, 3, 1, 2)
        self.d4 = DilatedConv(n, n, 3, 1, 4)
        self.d8 = DilatedConv(n, n, 3, 1, 8)
        self.d16 = DilatedConv(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4],1)
        output = self.bn(combine)
        output = self.act(output)
        return output




class DepthwiseESP(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = max(int(nOut/5),1)
        n1 = max(nOut - 4*n,1)
        self.c1 = DepthwiseSeparableConv(nIn, n, 1, 1)
        self.d1 = DepthwiseSeparableConv(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = DepthwiseSeparableConv(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = DepthwiseSeparableConv(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = DepthwiseSeparableConv(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = DepthwiseSeparableConv(n, n, 3, 1, 16) # dilation rate of 2^4
        self.bn = BatchnormRelu(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        #merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output

class AvgDownsampler(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''
    def __init__(self, config, use_swin=True):
        super().__init__()
        chanel_img = cfg.chanel_img
        model_cfg = cfg.sc_ch_dict[config] 
        self.use_swin = use_swin
        
        self.level1 = ConvBatchnormRelu(chanel_img, model_cfg['chanels'][0], stride = 2)
        self.sample1 = AvgDownsampler(1)
        self.sample2 = AvgDownsampler(2)

        self.b1 = ConvBatchnormRelu(model_cfg['chanels'][0] + chanel_img,model_cfg['chanels'][1])
        self.level2_0 = StrideESP(model_cfg['chanels'][1], model_cfg['chanels'][2])

        self.level2 = nn.ModuleList()
        for i in range(0, model_cfg['p']):
            self.level2.append(DepthwiseESP(model_cfg['chanels'][2] , model_cfg['chanels'][2]))
        self.b2 = ConvBatchnormRelu(model_cfg['chanels'][3] + chanel_img,model_cfg['chanels'][3] + chanel_img)

        # Stage 2 后的 Transformer Block
        if self.use_swin:
            # Stage 2 输出尺寸: 1/4 输入尺寸，假设输入为384x640，则输出为96x160
            self.swin_stage2 = create_swin_adapter(
                config=config,
                input_hw=(96, 160),  # Stage 2 输出尺寸
                use_dynamic=False,
                stage=2
            )
            # Stage 2 跳跃特征与Transformer融合模块
            self.skip_transformer_fusion2 = SkipTransformerFusion(
                skip_channels=chanel_img,  # inp1的通道数
                transformer_channels=model_cfg['chanels'][2],  # Stage 2输出通道数
                out_channels=model_cfg['chanels'][2],
                norm_layer=nn.BatchNorm2d
            )
            # Stage 2 LAFAM模块 - 融合跳跃特征、低分辨率特征和高分辨率特征
            self.lafam_stage2 = LAFAMWithSkip(
                in_ch_skip=chanel_img,  # inp1跳跃特征通道数
                in_ch_low=model_cfg['chanels'][2],  # Stage 2输出通道数
                in_ch_high=model_cfg['chanels'][2],  # Stage 2输出通道数
                out_ch=model_cfg['chanels'][2]  # 输出通道数
            )
        else:
            self.swin_stage2 = None
            self.skip_transformer_fusion2 = None
            self.lafam_stage2 = None

        self.level3_0 = StrideESP(model_cfg['chanels'][3] + chanel_img, model_cfg['chanels'][3])
        self.level3 = nn.ModuleList()
        for i in range(0, model_cfg['q']):
            self.level3.append(DepthwiseESP(model_cfg['chanels'][3] , model_cfg['chanels'][3]))
        
        # Stage 3 后的 Transformer Block
        if self.use_swin:
            # Stage 3 输出尺寸: 1/8 输入尺寸，假设输入为384x640，则输出为48x80
            self.swin_stage3 = create_swin_adapter(
                config=config,
                input_hw=(48, 80),  # Stage 3 输出尺寸
                use_dynamic=False,
                stage=3
            )
            # Stage 3 跳跃特征与Transformer融合模块
            self.skip_transformer_fusion3 = SkipTransformerFusion(
                skip_channels=chanel_img,  # inp2的通道数
                transformer_channels=model_cfg['chanels'][3],  # Stage 3输出通道数
                out_channels=model_cfg['chanels'][3],
                norm_layer=nn.BatchNorm2d
            )
            # Stage 3 LAFAM模块 - 融合跳跃特征、低分辨率特征和高分辨率特征
            self.lafam_stage3 = LAFAMWithSkip(
                in_ch_skip=chanel_img,  # inp2跳跃特征通道数
                in_ch_low=model_cfg['chanels'][3],  # Stage 3输出通道数
                in_ch_high=model_cfg['chanels'][3],  # Stage 3输出通道数
                out_ch=model_cfg['chanels'][3]  # 输出通道数
            )
        else:
            self.swin_stage3 = None
            self.skip_transformer_fusion3 = None
            self.lafam_stage3 = None
            
        self.b3 = ConvBatchnormRelu(model_cfg['chanels'][4],model_cfg['chanels'][2])
        
    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        # Stage 2 后应用 Transformer Block 和特征融合
        if self.use_swin and self.swin_stage2 is not None:
            B, C, H, W = output1.shape
            if H == 96 and W == 160:  # 检查Stage 2输出尺寸
                transformer_feat2 = self.swin_stage2(output1)
                # 融合跳跃特征inp1和Transformer特征
                if self.skip_transformer_fusion2 is not None:
                    skip_transformer_feat2 = self.skip_transformer_fusion2(inp1, transformer_feat2)
                else:
                    skip_transformer_feat2 = transformer_feat2
                
                # 使用LAFAM进一步融合跳跃特征、原始特征和Transformer特征
                if self.lafam_stage2 is not None:
                    output1 = self.lafam_stage2(inp1, output1, skip_transformer_feat2)
                else:
                    output1 = skip_transformer_feat2
            else:
                print(f"Warning: Stage 2 Swin adapter expects 96x160 input, got {H}x{W}. Skipping Swin processing.")

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        
        # Stage 3 后应用 Transformer Block 和特征融合
        if self.use_swin and self.swin_stage3 is not None:
            B, C, H, W = output2.shape
            if H == 48 and W == 80:  # 检查Stage 3输出尺寸
                transformer_feat3 = self.swin_stage3(output2)
                # 融合跳跃特征inp2和Transformer特征
                if self.skip_transformer_fusion3 is not None:
                    skip_transformer_feat3 = self.skip_transformer_fusion3(inp2, transformer_feat3)
                else:
                    skip_transformer_feat3 = transformer_feat3
                
                # 使用LAFAM进一步融合跳跃特征、原始特征和Transformer特征
                if self.lafam_stage3 is not None:
                    output2 = self.lafam_stage3(inp2, output2, skip_transformer_feat3)
                else:
                    output2 = skip_transformer_feat3
            else:
                print(f"Warning: Stage 3 Swin adapter expects 48x80 input, got {H}x{W}. Skipping Swin processing.")
        
        output2_cat=torch.cat([output2_0, output2], 1)
        out_encoder = self.b3(output2_cat)
        
        return out_encoder, output1, output2, inp1, inp2

class UpSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='prelu'):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        if activation == 'prelu':
            self.act = nn.PReLU(out_channels)
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1)
        elif activation == 'mish':
            self.act = Mish()

    def forward(self, input):
        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sub_dim=3, last=False, kernel_size=3, activation='prelu'):
        super(UpConvBlock, self).__init__()
        self.last=last
        self.up_conv = UpSimpleBlock(in_channels, out_channels, activation=activation)
        if not last:
            self.conv1 = ConvBatchnormRelu(out_channels+sub_dim,out_channels,kernel_size, activation=activation)
        self.conv2 = ConvBatchnormRelu(out_channels,out_channels,kernel_size, activation=activation)

    def forward(self, x, ori_img=None):
        x = self.up_conv(x)
        if not self.last:
            x = torch.cat([x, ori_img], dim=1)
            x = self.conv1(x)
        x = self.conv2(x)
        return x

class TwinLiteNetPlus(nn.Module):
    '''
    This class defines the ESPNet network
    '''

    def __init__(self, args=None):

        super().__init__()
        chanel_img = cfg.chanel_img
        model_cfg = cfg.sc_ch_dict[args.config] 
        
        # 获取use_swin参数
        self.use_swin = getattr(args, 'use_swin', True)
        
        # 获取use_lafam参数，默认为True
        self.use_lafam = getattr(args, 'use_lafam', True)
        
        # 将use_swin参数传递给Encoder
        self.encoder = Encoder(args.config, use_swin=self.use_swin)
        self.single_ob = False
        self.single_road = False

        # 全局LAFAM用于融合encoder的output1和output2
        if self.use_lafam:
            # output1: Stage 2输出，output2: Stage 3输出
            # 需要将output2上采样到output1的尺寸进行融合
            self.global_lafam = LAFAMWithSkip(
                in_ch_skip=chanel_img,  # 跳跃特征通道数
                in_ch_low=cfg.sc_ch_dict[args.config]['chanels'][3],  # output2通道数  
                in_ch_high=cfg.sc_ch_dict[args.config]['chanels'][2],  # output1通道数
                out_ch=cfg.sc_ch_dict[args.config]['chanels'][2]  # 输出通道数
            )
        
        self.caam = CAAM(feat_in=cfg.sc_ch_dict[args.config]['chanels'][2], num_classes=cfg.sc_ch_dict[args.config]['chanels'][2],bin_size =(2,4), norm_layer=nn.BatchNorm2d)
        self.conv_caam = ConvBatchnormRelu(cfg.sc_ch_dict[args.config]['chanels'][2],cfg.sc_ch_dict[args.config]['chanels'][1], activation='leakyrelu')

        self.up_1_road = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][1],cfg.sc_ch_dict[args.config]['chanels'][0], activation='leakyrelu') # out: Hx4, Wx4
        self.up_2_road = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][0],8, activation='leakyrelu') #out: Hx2, Wx2
        self.out_road = UpConvBlock(8,2,last=True, activation='leakyrelu')  

        self.up_1_ob = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][1],cfg.sc_ch_dict[args.config]['chanels'][0], activation='mish') # out: Hx4, Wx4
        self.up_2_ob = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][0],8, activation='mish') #out: Hx2, Wx2
        self.out_ob = UpConvBlock(8,cfg.obstacle_classes,last=True, activation='mish')


    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        out_encoder, output1, output2, inp1, inp2 = self.encoder(input)
        #visualize_feature_map_subset(out_encoder, "outencoder", 128)

        # 使用全局LAFAM融合encoder的output1和output2
        if self.use_lafam and hasattr(self, 'global_lafam'):
            # output1: Stage 2输出 (1/4尺寸), output2: Stage 3输出 (1/8尺寸)
            # LAFAM会将output2上采样到output1的尺寸进行融合
            fused_features = self.global_lafam(inp1, output2, output1)
            # 将融合后的特征下采样到out_encoder的尺寸，然后与原始encoder输出结合
            fused_features_downsampled = F.interpolate(fused_features, size=out_encoder.shape[2:], 
                                                    mode="bilinear", align_corners=False)
            out_encoder = out_encoder + fused_features_downsampled

        out_caam=self.caam(out_encoder)
        out_caam=self.conv_caam(out_caam)

        out_road=self.up_1_road(out_caam,inp2)
        out_road=self.up_2_road(out_road,inp1)
        out_road=self.out_road(out_road)

        out_ob=self.up_1_ob(out_caam,inp2)
        out_ob=self.up_2_ob(out_ob,inp1)
        out_ob=self.out_ob(out_ob)


        return out_road,out_ob

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])
import time

# 添加 Mish 激活函数
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class LAFAMWithSkip(nn.Module):
    """
    LAFAM with Skip Feature Guidance
    集成跳跃特征引导的轻量化自适应特征对齐模块
    """
    def __init__(self, in_ch_skip, in_ch_low, in_ch_high, out_ch):
        super(LAFAMWithSkip, self).__init__()
        
        # 跳跃特征通道对齐
        self.skip_proj = nn.Sequential(
            nn.Conv2d(in_ch_skip, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch)
        )
        
        # 原始LAFAM低分辨率分支 - 使用depthwise separable conv
        self.low_proj = nn.Sequential(
            nn.Conv2d(in_ch_low, in_ch_low, 3, padding=1, groups=in_ch_low, bias=False),
            nn.Conv2d(in_ch_low, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch)
        )
        
        # 原始LAFAM高分辨率分支 - 使用depthwise separable conv
        self.high_proj = nn.Sequential(
            nn.Conv2d(in_ch_high, in_ch_high, 3, padding=1, groups=in_ch_high, bias=False),
            nn.Conv2d(in_ch_high, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch)
        )
        
        # 通道注意力 - 融合skip和高分辨率特征后计算
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 基于融合特征计算空间权重
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 最终融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch)
        )
        
        self.act = nn.ReLU(inplace=True)

    def forward(self, feat_skip, feat_low, feat_high):
        """
        Args:
            feat_skip: 跳跃特征 [B, C_skip, H_skip, W_skip]
            feat_low: 低分辨率特征 [B, C_low, H_low, W_low] 
            feat_high: 高分辨率特征 [B, C_high, H_high, W_high]
        Returns:
            fused_feat: 融合后的特征 [B, out_ch, H_high, W_high]
        """
        # Step 1: 预处理跳跃特征 - 通道对齐
        feat_skip_proj = self.skip_proj(feat_skip)
        
        # Step 2: 尺寸对齐 - 将低分辨率特征上采样到高分辨率特征尺寸
        feat_low_aligned = F.interpolate(feat_low, size=feat_high.shape[2:], 
                                       mode="bilinear", align_corners=False)
        
        # Step 3: 特征投影
        feat_low_proj = self.low_proj(feat_low_aligned)
        feat_high_proj = self.high_proj(feat_high)
        
        # Step 4: 跳跃特征尺寸对齐
        if feat_skip_proj.shape[2:] != feat_high.shape[2:]:
            feat_skip_aligned = F.interpolate(feat_skip_proj, size=feat_high.shape[2:], 
                                            mode="bilinear", align_corners=False)
        else:
            feat_skip_aligned = feat_skip_proj
        
        # Step 5: 计算注意力权重
        # 通道注意力：融合跳跃特征和高分辨率特征
        fusion_for_channel = torch.cat([feat_skip_aligned, feat_high_proj], dim=1)
        channel_weight = self.channel_attention(fusion_for_channel)
        
        # 空间注意力：基于融合特征计算
        fusion_for_spatial = feat_skip_aligned + feat_high_proj
        spatial_weight = self.spatial_attention(fusion_for_spatial)
        
        # Step 6: 加权融合
        # 使用跳跃特征引导的加权融合
        weighted_low = feat_low_proj * channel_weight
        weighted_skip = feat_skip_aligned * spatial_weight
        
        # 最终融合：高分辨率特征 + 加权的低分辨率特征 + 加权的跳跃特征
        fused_feat = feat_high_proj + weighted_low + weighted_skip
        
        # Step 7: 最终处理
        out = self.fusion_conv(fused_feat)
        
        return self.act(out)


