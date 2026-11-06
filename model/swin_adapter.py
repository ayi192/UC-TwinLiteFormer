import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer_v2 import SwinTransformerBlock
import numpy as np


class LiteSwinV2(nn.Module):
    """
    轻量 Swin-V2 适配器：接收 (B,C,H,W)，内部用 1 个 SwinV2 block 做全局上下文增强。
    - 固定 input_resolution (H,W) 初始化一次；若分辨率会变化，用 pad 或重建模块。
    """
    def __init__(self, channels: int, hw: tuple, window_size: int = 8, num_heads: int = 2, mlp_ratio: float = 2.0):
        super().__init__()
        H, W = hw
        self.H, self.W = H, W
        self.channels = channels
        
        # 确保H和W能被window_size整除，如果不能则进行padding
        self.pad_h = (window_size - H % window_size) % window_size
        self.pad_w = (window_size - W % window_size) % window_size
        self.actual_h = H + self.pad_h
        self.actual_w = W + self.pad_w
        
        self.proj_in = nn.Identity()  # 若 channels 与 dim 不一致，用 1x1 Conv 调整
        self.norm_in = nn.LayerNorm(channels)  # Swin 里用 LN，作用在最后一维
        
        # 使用实际的padded尺寸来初始化Swin block
        self.block = SwinTransformerBlock(
            dim=channels,
            input_resolution=(self.actual_h, self.actual_w),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,  # SW-MSA
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.,
            norm_layer=nn.LayerNorm,
            pretrained_window_size=0
        )
        self.norm_out = nn.LayerNorm(channels)
        self.proj_out = nn.Identity()  # 若后续模块通道变更，再加 1x1 Conv
        self.residual = True  # 残差，稳定且几乎不增算力

    def forward(self, x):
        # x: (B,C,H,W) with H==self.H, W==self.W
        B, C, H, W = x.shape
        assert C == self.channels, f"expect channels {self.channels} got {C}"
        
        # 如果需要padding
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode='reflect')
        
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, self.actual_h * self.actual_w, C)  # (B,H*W,C)
        x = self.norm_in(x)
        y = self.block(x)  # (B,H*W,C)
        y = self.norm_out(y)
        
        if self.residual:
            y = y + x
        
        y = y.view(B, self.actual_h, self.actual_w, C).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        
        # 移除padding
        if self.pad_h > 0 or self.pad_w > 0:
            y = y[:, :, :self.H, :self.W]
        
        y = self.proj_out(y)
        return y


class DynamicLiteSwinV2(nn.Module):
    """
    动态尺寸的轻量 Swin-V2 适配器：支持任意输入尺寸
    """
    def __init__(self, channels: int, window_size: int = 8, num_heads: int = 2, mlp_ratio: float = 2.0):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.proj_in = nn.Identity()
        self.norm_in = nn.LayerNorm(channels)
        self.norm_out = nn.LayerNorm(channels)
        self.proj_out = nn.Identity()
        self.residual = True
        
        # 不预先创建block，在forward中动态创建

    def _create_swin_block(self, H, W):
        """动态创建Swin block"""
        return SwinTransformerBlock(
            dim=self.channels,
            input_resolution=(H, W),
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=self.window_size // 2,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.,
            norm_layer=nn.LayerNorm,
            pretrained_window_size=0
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels, f"expect channels {self.channels} got {C}"
        
        # 确保H和W能被window_size整除
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        actual_h = H + pad_h
        actual_w = W + pad_w
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, actual_h * actual_w, C)
        x = self.norm_in(x)
        
        # 动态创建block并移动到正确的设备
        block = self._create_swin_block(actual_h, actual_w)
        # 确保block在正确的设备上
        device = x.device
        block = block.to(device)
        y = block(x)
        y = self.norm_out(y)
        
        if self.residual:
            y = y + x
        
        y = y.view(B, actual_h, actual_w, C).permute(0, 3, 1, 2).contiguous()
        
        # 移除padding
        if pad_h > 0 or pad_w > 0:
            y = y[:, :, :H, :W]
        
        y = self.proj_out(y)
        return y


def create_swin_adapter(config: str, input_hw: tuple, use_dynamic: bool = False, stage: int = 3):
    """
    根据配置创建Swin适配器
    
    Args:
        config: 模型配置 ('nano', 'small', 'medium', 'large')
        input_hw: 输入特征的高度和宽度 (H, W)
        use_dynamic: 是否使用动态尺寸版本
        stage: Stage编号 (2 或 3)
    
    Returns:
        Swin适配器实例
    """
    from . import config as cfg
    
    # 根据配置和Stage确定通道数
    model_cfg = cfg.sc_ch_dict[config]
    if stage == 2:
        channels = model_cfg['chanels'][2]  # Stage 2 通道数
    elif stage == 3:
        channels = model_cfg['chanels'][3]  # Stage 3 通道数
    else:
        channels = model_cfg['chanels'][2]  # 默认使用Stage 2通道数
    
    # 根据配置调整参数
    if config == 'nano':
        window_size = 8
        num_heads = 2
        mlp_ratio = 2.0
    elif config == 'small':
        window_size = 8
        num_heads = 4
        mlp_ratio = 2.5
    elif config == 'medium':
        window_size = 8
        num_heads = 4
        mlp_ratio = 3.0
    else:  # large
        window_size = 8
        num_heads = 8
        mlp_ratio = 3.0
    
    if use_dynamic:
        return DynamicLiteSwinV2(
            channels=channels,
            window_size=window_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
    else:
        return LiteSwinV2(
            channels=channels,
            hw=input_hw,
            window_size=window_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
