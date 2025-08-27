import torch
import torch.nn as nn
from einops import rearrange, repeat
from opencood.models.my_method.residual_compress import psm_query
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor

import time

from opencood.models.sub_modules.fuse_utils import regroup
import torch.nn.functional as F
import numpy as np
import torch
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

from opencood.models.base_transformer import FeedForward, PreNormResidual
from torch.nn.functional import interpolate as interpolate
import math
from opencood.tools.visualize_heatmap import feature_heatmap,visualize_batch_heatmaps,residul_heatmap,aggrate_heatmap
from opencood.tools.my_tools import cal_size

from opencood.models.my_method.visualization import PointCloudVisualizer

def get_selected_cav_feature(x, record_len,selected_cav_id_list):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    out = []
    idx = 0
    for xx in split_x:
        xx = xx[selected_cav_id_list[idx]].unsqueeze(0)
        out.append(xx)
        idx = idx + 1 
    return torch.cat(out, dim=0)

def get_ego_feature(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    out = []
    for xx in split_x:
        xx = xx[0].unsqueeze(0)
        out.append(xx)
    return torch.cat(out, dim=0)

def get_fused_ego_feature(x):
    B,N,C,H,W = x.shape
    out = []
    for b in range(B):
       xx = x[b][0].unsqueeze(0)
       out.append(xx)
    return torch.cat(out, dim=0)



class residual_aggrate_filter(nn.Module):
    def __init__(self, args):
        super(residual_aggrate_filter, self).__init__()
        
        self.threshold=args['threshold']
        self.feature_dim = args['anchor_nums']  # 特征图的通道数
        self.dim=args['dim']
        self.comm_volume=args['comm_volume']
        
        self.residual_weight=1
        self.aggrate_weight=1
        
        
    def generte_mask(self,score,k):
        # score 为置信图的分数，大小为 1*h*w
        
        # 获取前 k 个最大的元素
        threshold_value = torch.topk(score.view(-1), k, sorted=False).values.min()
        # 构造掩码
        filter_mask = (score >= threshold_value).float()
        
        return filter_mask
        

    def forward(self,ego_psm,cav_psm,flag):
       
        C, H, W = ego_psm.shape
        residual_psm=cav_psm-ego_psm
        aggrate_psm=cav_psm+ego_psm
        
        residual_psm,_=residual_psm.sigmoid().max(dim=0, keepdim=True)
        aggrate_psm,_=aggrate_psm.sigmoid().max(dim=0, keepdim=True)
        
        
        # 获取应该保留的元素个数
        if self.training:
            k_percent= torch.rand(1)
            num_elements = residual_psm.numel()  # 张量中的总元素数
            k = max(1, int(num_elements * k_percent))  # 确保至少保留 1 个元素

        elif self.threshold:
            k_percent=self.threshold
            num_elements = residual_psm.numel()  # 张量中的总元素数
            k = max(1, int(num_elements * k_percent))  # 确保至少保留 1 个元素
        else:
            k=min(int(2**self.comm_volume/4/self.dim),residual_psm.numel())
        
        #划分两种策略的元素个数
        filter_residual=self.generte_mask(residual_psm,max(1,int(k*self.residual_weight)))
        
        filter_aggrate=self.generte_mask(aggrate_psm,max(1,int(k*self.aggrate_weight)))
        
        filter_mask,_=torch.concatenate([filter_aggrate,filter_residual],dim=0).max(dim=0,keepdim=True)

        # if flag:
        #     residul_heatmap(filter_residual)
        # feature_heatmap(filter_mask)


        return filter_mask,filter_residual,filter_aggrate
        
        


class psm_query(nn.Module):
    def __init__(self, args):
        super(psm_query, self).__init__()
        
        self.method=args['method']

        self.filter= residual_aggrate_filter(args)
            
    def forward(self,x,psm,mask,flag):
        B,L,C,H,W=x.shape
        
        filter_feature_list=[]
        residual_feature_list=[]
        aggrate_feature_list=[]
        for b in range(B):
            ego_psm=psm[b][0]
            batch_feature=[]
            residual_feature=[]
            aggrate_feature=[]
            for i in range(L):
                # ego 车辆无需过滤
                if i==0 :
                    batch_feature.append(x[b][i])
                    residual_feature.append(x[b][i])
                    aggrate_feature.append(x[b][i])
                
                # 没有协同车辆
                elif mask[b][i]==0 :
                    batch_feature.append(torch.zeros([C,H,W]).to(x.device))
                    residual_feature.append(torch.zeros([C,H,W]).to(x.device))
                    aggrate_feature.append(torch.zeros([C,H,W]).to(x.device))
        
                else:
                    cav_psm=psm[b][i]
                    filter_mask,residual_mask,aggrate_mask=self.filter(ego_psm,cav_psm,flag)
                    batch_feature.append(x[b][i]*filter_mask)
                    residual_feature.append(x[b][i]*residual_mask)
                    aggrate_feature.append(x[b][i]*aggrate_mask)
                    
            filter_feature_list.append(torch.stack(batch_feature,dim=0))
            residual_feature_list.append(torch.stack(residual_feature,dim=0))
            aggrate_feature_list.append(torch.stack(aggrate_feature,dim=0))
            
        filter_feature=torch.stack(filter_feature_list,dim=0)
        residual_feature=torch.stack(residual_feature_list,dim=0)
        aggrate_feature=torch.stack(aggrate_feature_list,dim=0)
                    

        return filter_feature,residual_feature,aggrate_feature


class SE(nn.Module):
    """
    Squeeze and excitation block
    """

    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
class FeatExtract(nn.Module):
    """
    Feature extraction block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """
    def __init__(self, dim,strides,keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=strides, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x
    
class WindowAttentionGlobal(nn.Module):
    """
    Global window attention based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """
    def __init__(
            self,
            dim = 256,
            dim_head=32,
            dropout=0.,
            agent_size=2,
            window_size=7
    ):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """
        assert (dim % dim_head) == 0, \
            'dimension should be divisible by dimension per head'
        self.head_dim = dim_head
        self.num_heads = dim // dim_head
        self.scale =  dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]
        self.agent_size = agent_size
        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1),
            self.num_heads)  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 3, Wd, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= \
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 2)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.to_q_global = nn.Sequential(
                FeatExtract(dim,(2,2),keep_dim=False),
                FeatExtract(dim,(2,2), keep_dim=False),
                # FeatExtract(dim, keep_dim=False),
            )
        # self.conv3d = nn.Conv3d(self.max_cav, 1, 1, stride=1, padding=0)
    def forward(self, x,mask=None):
        batch, agent_size, height, width, window_height, window_width, _, device, h \
            = *x.shape, x.device, self.num_heads
        # x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4)
        #[1，2，15，45，4，4，256]
        # print(x.shape) B*L C H W
        x_global = rearrange(x, 'b m x y w1 w2 d -> (b m) d (x w1) (y w2)') # 为了后面使用卷积进行下采样

        x_global = self.to_q_global(x_global) # H和W的大小都除以了win_size，使用卷积下采样得到的结果
        B, C, H, W = x_global.shape # batch*N

        x_global = rearrange(x_global, '(b m) d (x w1) (y w2) -> b m x y w1 w2 d',
                      m=agent_size,w1=window_height, w2=window_height) # 对下采样后的特征再划分窗口
        x_global = rearrange(x_global, 'b l x y w1 w2 d -> b (x y) (l w1 w2) d')
        g_attn = (x_global @ x_global.transpose(-2, -1))
        g_attn = self.softmax(g_attn)*(C**-0.5)

        x_global = einsum('b h i j, b h j d -> b h i d', g_attn, x_global)

        q_global = torch.mean(x_global, dim=1, keepdim=True) # 将所有窗户的特征平均化
        q_global = q_global.reshape(batch,1,agent_size*window_height*window_height,self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4) # 划分多头 B*1*(l w1 w2)*H*C
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        B_, N, C = x.shape
        B = q_global.shape[0]
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor') #每个头的维度
        B_dim = torch.div(B_, B, rounding_mode='floor') #每个batch包含窗口的个数
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4) #2是k和v
        k, v = kv[0], kv[1]
        q_global = q_global.repeat(1, B_dim, 1, 1, 1) #综合了之前的窗口注意力，复制给了每个窗口，每个窗口的q都包含全局特征
        q = q_global.reshape(B_, self.num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))


        relative_position_bias = self.relative_position_bias_table(self.relative_position_index)
        # sim = sim + rearrange(bias, 'i j h -> h i j')
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # print(attn.shape)
        # print(relative_position_bias.shape)
        attn = attn + relative_position_bias.unsqueeze(0)
        # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # [675,1,1,32]
            # b x y w1 w2 e l -> (b x y) 1 (l w1 w2)
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -float('inf'))

        attn = self.softmax(attn)
        # attn = self.attn_drop(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
                        l=agent_size, w1=window_height, w2=window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d',
                         b=batch, x=height, y=width)




class DotAttention(nn.Module):
    def __init__(
            self,
            dim = 256,
            dim_head=32,
            dropout=0.,
            agent_size=6,
            window_size=7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, \
            'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1),
            self.heads)  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 3, Wd, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= \
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, x, mask=None):
        # x shape: B,L,H,W,C
        # [1,2,15,45,4,4,256]
        batch, agent_size, height, width, _,  device, h = *x.shape, x.device, self.heads

        x = rearrange(x, 'b l h w d -> (b h w) l d')

        # self.to_qkv(x) ->    C->3C
        # chunk -1 表示沿着最后一个维度进行分割
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        #'b' 仍然是批处理维度， 'h' 表示头（head）的数量， 'n' 表示车辆编号， 'd' 表示每个头的维度，!!!这里序列长度表示每个车的一个窗户所有像素!!!
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      (q, k, v))
        # scale
        q = q * self.scale

        # sim
        # [675,8,32,32]
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        

        # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # [675,1,1,32]
            # b x y w1 w2 e l -> (b x y) 1 (l w1 w2)
            mask = rearrange(mask, 'b x y e l -> (b x y) e l')
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            #使用 masked_fill 函数，将 sim 中与 mask 中的零值对应的位置的值替换为负无穷（-inf）。这意味着这些位置的注意力权重将被设置为零，即模型不会关注这些位置。
            # 这些位置表示没有智能体的地方
            sim = sim.masked_fill(mask == 0, -float('inf'))

        # attention  Softmax函数
        attn = self.attend(sim)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h l d -> b l (h d)',l=agent_size)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) l d -> b l x y d',b=batch, x=height, y=width)


class Attention(nn.Module):
    """
    Unit Attention class. Todo: mask is not added yet.

    Parameters
    ----------
    dim: int
        Input feature dimension.
    dim_head: int
        The head dimension.
    dropout: float
        Dropout rate
    agent_size: int
        The agent can be different views, timestamps or vehicles.
    """

    def __init__(
            self,
            dim = 256,
            dim_head=32,
            dropout=0.,
            agent_size=6,
            window_size=7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, \
            'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1),
            self.heads)  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 3, Wd, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= \
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, x, mask=None):
        # x shape: b, l, h, w, w_h, w_w, c
        # [1,2,15,45,4,4,256]
        batch, agent_size, height, width, window_height, window_width, _, device, h \
            = *x.shape, x.device, self.heads

        # flatten
        #[675(B*H*W/P^2),32,N*P^2,256 C]


        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        # project for queries, keys, values

        # self.to_qkv(x) ->    C->3C
        # chunk -1 表示沿着最后一个维度进行分割
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # split heads
        # [675,8,32,32]
        #'b' 仍然是批处理维度， 'h' 表示头（head）的数量， 'n' 表示序列长度维度， 'd' 表示每个头的维度，!!!这里序列长度表示每个车的一个窗户所有像素!!!
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      (q, k, v))
        # scale
        q = q * self.scale

        # sim
        # [675,8,32,32]
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        # [32,32,8]
        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # [675,1,1,32]
            # b x y w1 w2 e l -> (b x y) 1 (l w1 w2)
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            #使用 masked_fill 函数，将 sim 中与 mask 中的零值对应的位置的值替换为负无穷（-inf）。这意味着这些位置的注意力权重将被设置为零，即模型不会关注这些位置。
            # 这些位置表示没有智能体的地方
            sim = sim.masked_fill(mask == 0, -float('inf'))

        # attention  Softmax函数
        attn = self.attend(sim)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
                        l=agent_size, w1=window_height, w2=window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d',
                         b=batch, x=height, y=width)

class ERMVPFusionBlockMask(nn.Module):

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 dim_head,
                 window_size,
                 agent_size,
                 drop_out):
        super(ERMVPFusionBlockMask, self).__init__()

        self.window_size = window_size

        self.window_attention = PreNormResidual(input_dim,
                                                Attention(input_dim, dim_head,
                                                          drop_out,
                                                          agent_size,
                                                          window_size))
        self.window_ffd = PreNormResidual(input_dim,
                                          FeedForward(input_dim, mlp_dim,
                                                      drop_out))
        
        self.point_fused=PreNormResidual(input_dim,DotAttention(input_dim, dim_head,drop_out,agent_size))
        
        self.point_ffd = PreNormResidual(input_dim,
                                          FeedForward(input_dim, mlp_dim,
                                                      drop_out))
        

    def forward(self, x, mask):
        # x: b l c h w,出来也是这个格式
        # mask: b h w 1 l
        # window attention -> grid attention

        B,L,C,H,W=x.shape
        x=rearrange(x,'b l c h w -> b l h w c')
        mask_swap=mask
    
        x=self.point_fused(x,mask=mask_swap)
        x=self.point_ffd(x)

        # mask b h w 1 l -> b x y w1 w2 1 L
        # [1,15,45,4,4,1,2]
        mask_swap = rearrange(mask_swap,
                              'b (x w1) (y w2) e l -> b x y w1 w2 e l',
                              w1=self.window_size, w2=self.window_size)
        #[1，2，15，45，4，4，256]
        x = rearrange(x, 'b m (x w1) (y w2) d -> b m x y w1 w2 d',
                      w1=self.window_size, w2=self.window_size)
        #[1，2，15，45，4，4，256]
        x = self.window_attention(x, mask=mask_swap)
        x = self.window_ffd(x)
        #[1,2,256,60,180]
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')
        
        return x

class ERMVPFusionBlock(nn.Module):

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 dim_head,
                 window_size,
                 agent_size,
                 drop_out):
        super(ERMVPFusionBlock, self).__init__()
        # b = batch * max_cav
        self.block = nn.Sequential(
            Rearrange('b m d (x w1) (y w2) -> b m x y w1 w2 d',
                      w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out,
                                                 agent_size, window_size)),
            PreNormResidual(input_dim,
                            FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange('b m x y w1 w2 d -> b m d (x w1) (y w2)'),

            Rearrange('b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                      w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out,
                                                 agent_size, window_size)),
            PreNormResidual(input_dim,
                            FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange('b m x y w1 w2 d -> b m d (w1 x) (w2 y)'),
        )

    def forward(self, x, mask=None):
        # todo: add mask operation later for mulit-agents
        x = self.block(x)
        return x

class ERMVPFusionEncoder(nn.Module):

    def __init__(self, args):
        super(ERMVPFusionEncoder, self).__init__()

        self.layers = nn.ModuleList([])
        self.depth = args['depth']

        # block related
        input_dim = args['input_dim']
        mlp_dim = args['mlp_dim']
        agent_size = args['agent_size']
        window_size = args['window_size']
        drop_out = args['drop_out']
        dim_head = args['dim_head']

        self.mask = False
        if 'mask' in args:
            self.mask = args['mask']

        for i in range(self.depth):
            if self.mask:
                block = ERMVPFusionBlockMask(input_dim,
                                    mlp_dim,
                                    dim_head,
                                    window_size[i],
                                    agent_size,
                                    drop_out)

            else:
                block = ERMVPFusionBlock(input_dim,
                                        mlp_dim,
                                        dim_head,
                                        window_size,
                                        agent_size,
                                        drop_out)
            self.layers.append(block)

        # mlp head
        self.mlp_head = nn.Sequential(
            Reduce('b m d h w -> b d h w', 'mean'),
            Rearrange('b d h w -> b h w d'),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange('b h w d -> b d h w')
        )

    def forward(self, x, mask=None):
        for stage in self.layers:
            x = stage(x, mask=mask)
        return self.mlp_head(x)


class LateFusion(nn.Module):
    def __init__(self, args):
        super(LateFusion, self).__init__()
        # self.thre = args['thre']
        if 'gaussian_smooth' in args:
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()
        
    def forward(self,input,psm_residual,psm_aggrate):
        B,C,H,W = input[0].shape
        fusion_list = []
        # 对每一个特征乘了相应的置信图，加到一起
        for b in range(B):
            confi_residual = psm_residual[b:b+1,:].sigmoid().max(dim=1)[0].unsqueeze(1)  
            # confi_ego = psm_ego[b:b+1,:].sigmoid().max(dim=1)[0].unsqueeze(1)  
            confi_aggrate = psm_aggrate[b:b+1,:].sigmoid().max(dim=1)[0].unsqueeze(1)
            if self.smooth:
                confi_residual = self.gaussian_filter(confi_residual)
                # confi_ego = self.gaussian_filter(confi_ego)
                confi_aggrate = self.gaussian_filter(confi_aggrate)
            total_confi = torch.cat([confi_residual,confi_aggrate],dim=1)
            total_confi = torch.softmax(total_confi,dim=1)
            feat_residual = input[0][b:b+1,:] * total_confi[0:1,0:1,:,:]  
            feat_aggrate = input[1][b:b+1,:] * total_confi[0:1,1:2,:,:]  
            # feat_agent = input[2][b:b+1,:] * total_confi[0:1,2:3,:,:]  
            fusion_list.append(feat_residual + feat_aggrate)  
        final_feat = torch.cat(fusion_list,dim=0)  
        return final_feat




class PointPillarWhat2keep(nn.Module):
    def __init__(self, args):
        super(PointPillarWhat2keep, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        
        self.compress=psm_query(args['compress'])

        self.residual_fusion = ERMVPFusionEncoder(args['ermvp_fusion'])
        
        self.aggrate_fusion = ERMVPFusionEncoder(args['ermvp_fusion'])

        # self.vehicle_fusion = ERMVPFusionEncoder(args['ermvp_fusion'])
        
        # self.fusion_net=ERMVPFusionEncoder(args['ermvp_fusion'])
        
        self.late_fusion=LateFusion(args=args['late_fusion'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        
        self.vis_featuremap=False

        if args['backbone_fix']:
            self.backbone_fix()        
        


    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def prepare_vis(self,data_dict):
        lidar_np=data_dict['lidar_np']
        non_delay_lidar_np=data_dict['non_delay_lidar_np']

        # 无时延：
        voxel_features = data_dict['non_delay_processed_lidar']['voxel_features']
        voxel_coords = data_dict['non_delay_processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['non_delay_processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # [1,384,120,360]
        non_delay_spatial_features_2d = batch_dict['spatial_features_2d']

        # 有时延：
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # [1,384,120,360]
        spatial_features_2d = batch_dict['spatial_features_2d']

        
        self.visualize=PointCloudVisualizer(non_delay_lidar_np, lidar_np, non_delay_spatial_features_2d , spatial_features_2d )

    def forward(self, data_dict, latency=False):
        # latency为True是对无时延的推理，False是正常有时延的推理
        # data_dict['prior_encoding'][0][:,1] 为5辆车每个车辆的时延，以100ms为单位

        if latency:
            self.prepare_vis(data_dict)
            voxel_features = data_dict['non_delay_processed_lidar']['voxel_features']
            voxel_coords = data_dict['non_delay_processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['non_delay_processed_lidar']['voxel_num_points']
        else:
            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # [1,384,120,360]
        spatial_features_2d = batch_dict['spatial_features_2d']


        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # [1,384,60,180]

        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        ego_psm = self.cls_head(spatial_features_2d)

        regroup_feature, mask = regroup(spatial_features_2d,record_len,self.max_cav)
        
        regroup_psm , _=regroup(ego_psm,record_len,self.max_cav)
        
        batch_feature,residual_feature,aggrate_feature=self.compress(regroup_feature,regroup_psm,mask,self.vis_featuremap)
        
        non_zero_mask = torch.any(batch_feature[0,1] != 0, dim=0)
        psm_map = non_zero_mask.to(torch.int).unsqueeze(0).to(torch.float)
        # feature_heatmap(psm_map)
        
        # [1,1,1,1,2]
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # [1,60,180,1,2]
        com_mask = repeat(com_mask,
                          'b h w c l -> b (h new_h) (w new_w) c l',
                          new_h=batch_feature.shape[3],
                          new_w=batch_feature.shape[4])
        
        residual_fused = self.aggrate_fusion(residual_feature,com_mask)
        
        # late_fusion
        aggrate_fused = self.residual_fusion(aggrate_feature,com_mask)


        # aggrate_fused = self.vehicle_fusion(aggrate_feature,com_mask)
        # residual_fused = self.vehicle_fusion(aggrate_feature,com_mask)

        residual_psm=self.cls_head(residual_fused)
        aggrate_psm=self.cls_head(aggrate_fused)
        fused_feature=self.late_fusion([residual_fused,aggrate_fused],residual_psm,aggrate_psm)

        # only residual
        # fused_feature=residual_fused

        
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm,
                       'comm_volume':cal_size(batch_feature[0,1,...]) if (not self.training) and record_len[0]>1 else None
                       }


        return output_dict

