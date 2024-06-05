"""EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2023)
Xin Liu, Brial Hill, Ziheng Jiang, Shwetak Patel, Daniel McDuff
"""
import numpy as np
import torch
import torch.nn as nn
import cv2
from einops import rearrange
from neural_methods.model.video_swin_transformer import *
class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, -1, :fold] = x[:, 0, :fold] # wrap left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold]  # wrap right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # no shift for final fold
        return out.view(nt, c, h, w)

class VSWM(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20, img_size=36, channel='raw'
                 ,patch_size=(4,4,4),in_chans=3,embed_dim=96,norm_layer=nn.LayerNorm,depths=[2, 2, 6, 2],
                 patch_norm=False,drop_rate = 0.,num_heads=[3, 6, 12, 24],window_size=(2,7,7),mlp_ratio=4.,
                 qkv_bias=True,qk_scale=None,drop=0,attn_drop_rate=0., drop_path_rate=0.2,use_checkpoint=False
                 ):
        super(VSWM, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                  bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                  bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)

        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        elif img_size == 224:
            self.final_dense_1 = nn.Linear(6912, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        self.num_layers = len(depths)
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)
        self.final_dense_3 = nn.Linear(self.nb_dense, 4, bias=True)
        self.batch_norm = nn.BatchNorm2d(3)
        self.channel = channel
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_size = patch_size
        self.norm = norm_layer(self.num_features)
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    def forward(self, inputs, params=None):
        # inputs = torch.diff(inputs, dim=0)
        # 对输入batch的每一个特征通道进行normalize
        # inputs = self.batch_norm(inputs)
        nt,c, H, W=inputs.shape
        network_input=inputs.unsqueeze(1)
        D,B,C,H, W=network_input.shape

        network_input=network_input.view(B, C, D, H, W)
        x = self.patch_embed(network_input)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x.contiguous())
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = rearrange(x,' n d h w c -> (n d) c h w')

        d9 = x.reshape(x.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        d12 =self.final_dense_3(d11)
        out = d12.reshape(D).unsqueeze(1)

        return out
