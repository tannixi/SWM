"""EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2023)
Xin Liu, Brial Hill, Ziheng Jiang, Shwetak Patel, Daniel McDuff
"""
import numpy as np
import torch
import torch.nn as nn
import cv2
from neural_methods.model.swin_transformer import *

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
        # nt为帧数，h，w输入特征图大小，c为特征数
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

class SWM(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=30, frame_depth=20, img_size=36, channel='raw',mlp_ratio=4.,
                 patch_size=4, in_chans=3,embed_dim=96,norm_layer=nn.LayerNorm,ape=False,drop_rate=0., depths=[2, 2, 6, 2]
                 , num_heads=[3, 6, 12, 24], window_size=8,qkv_bias=True,qk_scale=None, attn_drop_rate=0., drop_path_rate=0.1,use_checkpoint=False
                 ):
        super(SWM, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = 128
        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Dropout layers
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        elif img_size == 224:
            self.final_dense_1 = nn.Linear(9216, self.nb_dense, bias=True)
        elif img_size == 128:
            self.final_dense_1 = nn.Linear(12288, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)
        self.batch_norm = nn.BatchNorm2d(3)
        self.channel = channel
        self.ape = False
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer )
        num_patches = self.patch_embed.num_patches
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.ape = ape
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.layers = nn.ModuleList()
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
            self.norm = norm_layer(self.num_features)




    def forward(self, inputs, params=None):

        # frames = inputs.cpu().numpy()
        # frame=frames[0]
        # C, H, W = frame.shape
        # frame = frame.transpose((1,2,0))
        # b, g, r = cv2.split(frame)  # 分别提取B、G、R通道
        # frame = cv2.merge([r, g, b])  # 重新组合为R、G、B
        # save_name = '/data/tannixi_datas/feature_map/' + 'inputs' + '.png'
        # cv2.imwrite(save_name, frame*255)
        # cv2.waitKey(0)

        inputs = torch.diff(inputs, dim=0)

        # frames = inputs.cpu().numpy()
        # frame=frames[0]
        # C, H, W = frame.shape
        # frame = frame.transpose((1,2,0))
        # b, g, r = cv2.split(frame)  # 分别提取B、G、R通道
        # frame = cv2.merge([r, g, b])  # 重新组合为R、G、B
        # save_name = '/data/tannixi_datas/feature_map/' + 'diff' + '.png'
        # cv2.imwrite(save_name, frame*255)
        # cv2.waitKey(0)

        # 对输入batch的每一个特征通道进行normalize
        inputs = self.batch_norm(inputs)

        # frames = inputs.cpu().detach().numpy()
        # frame=frames[0]
        # C, H, W = frame.shape
        # frame = frame.transpose((1,2,0))
        # b, g, r = cv2.split(frame)  # 分别提取B、G、R通道
        # frame = cv2.merge([r, g, b])  # 重新组合为R、G、B
        # save_name = '/data/tannixi_datas/feature_map/' + 'batch_norm' + '.png'
        # cv2.imwrite(save_name, frame*255)
        # cv2.waitKey(0)


        # network_input = self.TSM_1(inputs)

        # frames = network_input.cpu().detach().numpy()
        # frame=frames[0]
        # C, H, W = frame.shape
        # frame = frame.transpose((1,2,0))
        # b, g, r = cv2.split(frame)  # 分别提取B、G、R通道
        # frame = cv2.merge([r, g, b])  # 重新组合为R、G、B
        # save_name = '/data/tannixi_datas/feature_map/' + 'TSM_1' + '.png'
        # cv2.imwrite(save_name, frame*255)
        # cv2.waitKey(0)

        # 做一个patch_embed操作
        network_input = self.patch_embed(inputs)
        if self.ape:
            network_input = network_input + self.absolute_pos_embed
        x = self.pos_drop(network_input)

        # 做一个layer操作
        for layer in self.layers:
            x = layer(x)

        d9 = x.view(x.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return out