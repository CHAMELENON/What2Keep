import torch.nn as nn
import torch
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.v2v_fuse import V2VNetFusion
# from opencood.tools.visualize_heatmap import visualize_batch_heatmaps,feature_heatmap
import torch.nn.functional as F
# from opencood.tools.visualize_heatmap import visualize_batch_heatmaps,feature_heatmap
# from opencood.tools.my_tools import cal_size

from opencood.tools.visualize_heatmap import feature_heatmap,visualize_batch_heatmaps

class attention_mask(nn.Module):
    def __init__(self, args):
        super(attention_mask, self).__init__()
        
        self.threshold=args['threshold']
        self.feature_dim = args['anchor_nums']  # 特征图的通道数
        self.dim=args['dim']
        self.comm_volume=args['comm_volume']
        
        # 可学习的线性层
        # self.query_transform = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        # self.key_transform = nn.Linear(self.feature_dim, self.feature_dim, bias=False)

    def forward(self,ego_psm,cav_psm):
        C, H, W = ego_psm.shape
        S = H * W

        # 展平特征图
        F_self_flat = ego_psm.view(C, S).unsqueeze(0).permute(0, 2, 1)
        F_agents_flat = cav_psm.view(C, S).unsqueeze(0).permute(0, 2, 1)
        
        # 通过线性层进行变换
        # F_self_flat = self.query_transform(F_self_flat)  # shape = (1,S, C)
        # F_agents_flat = self.key_transform(F_agents_flat)  # shape = (1,S, C)

        # 计算相似性矩阵
        sim_matrix = torch.bmm(F_self_flat,  # shape = (1, S, C)
                            F_agents_flat.transpose(1, 2))  # shape = (1, S, S)

        # 归一化 (softmax)
        attention_weights = F.softmax(sim_matrix / C**0.5, dim=-1)  # shape = (1, S, S)

        # 应用注意力掩码
        F_updated_flat,_ = torch.bmm(attention_weights, F_agents_flat).sigmoid().max(dim=2, keepdim=True) # shape = (1, S, C)
        F_updated_flat=F_updated_flat.permute(0,2,1).view(1,1,H,W)[0]
        
        if self.training:
            k_percent= torch.rand(1)
            num_elements = F_updated_flat.numel()  # 张量中的总元素数
            k = max(1, int(num_elements * k_percent))  # 确保至少保留 1 个元素

        elif self.threshold:
            k_percent=self.threshold
            num_elements = F_updated_flat.numel()  # 张量中的总元素数
            k = max(1, int(num_elements * k_percent))  # 确保至少保留 1 个元素
        else:
            k=min(int(2**self.comm_volume/4/self.dim),F_updated_flat.numel())
    
        # 获取前 k 个最大的元素
        threshold_value = torch.topk(F_updated_flat.view(-1), k, sorted=False).values.min()
        # 构造掩码
        filter_mask = (F_updated_flat >= threshold_value).float()
        
        return filter_mask
        
        




class conv_mask(nn.Module):
    def __init__(self, args):
        super(conv_mask, self).__init__()
        
        self.threshold=args['threshold']
            
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(args['anchor_nums'] * 2, args['output_dim'], kernel_size=5, padding=2),  # 第一个卷积层
            nn.Conv2d(args['output_dim'], 1, kernel_size=1)  # 第二个卷积层
        ])

    
    def forward(self,ego_psm,cav_psm):
        single_psm=torch.cat((ego_psm,cav_psm),dim=0).unsqueeze(0)
        filter=single_psm.clone()
        for conv in self.conv_layers:
            filter = conv(filter)
        #将l,c,h,w变成 c,h,w ,后面再按车辆叠加
        filter=filter[0]

        if self.training:
            k_percent=torch.rand(1)
        else:
            k_percent=self.threshold
            
        # 计算保留元素的数量
        num_elements = filter.numel()  # 张量中的总元素数
        k = max(1, int(num_elements * k_percent))  # 确保至少保留 1 个元素
        # 获取前 k 个最大的元素
        threshold_value = torch.topk(filter.view(-1), k, sorted=False).values.min()
        # 构造掩码
        filter_mask = (filter >= threshold_value).float()
        
        return filter_mask
        



class psm_query(nn.Module):
    def __init__(self, args):
        super(psm_query, self).__init__()
        
        self.method=args['method']
        
        self.attention_filter= attention_mask(args)
        
        self.conv_filter=conv_mask(args)
        
    def forward(self,x,psm,mask):
        B,L,C,H,W=x.shape
        
        filter_feature_list=[]
        for b in range(B):
            ego_psm=psm[b][0]
            batch_feature=[]
            for i in range(L):
                # ego 车辆无需过滤
                if i==0 :
                    batch_feature.append(x[b][i])
                
                # 没有协同车辆
                elif mask[b][i]==0 :
                    batch_feature.append(torch.zeros([C,H,W]).to(x.device))
        
                else:
                    cav_psm=psm[b][i]
                    if self.method=="conv":
                        filter_mask=self.conv_filter(ego_psm,cav_psm)
                    elif self.method=="attention":
                        filter_mask=self.attention_filter(ego_psm,cav_psm)
                        
                    batch_feature.append(x[b][i]*filter_mask)
                    
            filter_feature_list.append(torch.stack(batch_feature,dim=0))
            
        filter_feature=torch.stack(filter_feature_list,dim=0)
                    

        return filter_feature








# class psm_query(nn.Module):
#     def __init__(self, args):
#         super(psm_query, self).__init__()
        
#         if self.training:
#             self.threshold=torch.rand(1)
#         else:
#             self.threshold=args['threshold']
#         # 两两交互，输出一个置信度，用于过滤
#         self.conv_layers = nn.ModuleList([
#             nn.Conv2d(args['anchor_nums'] * 2, args['output_dim'], kernel_size=5, padding=2),  # 第一个卷积层
#             nn.Conv2d(args['output_dim'], 1, kernel_size=1)  # 第二个卷积层
#         ])

#     def forward(self,x,psm,mask):
#         B,L,C,H,W=x.shape
        
#         filter_feature_list=[]
#         for b in range(B):
#             ego_psm=psm[b][0]
#             batch_feature=[]
#             for i in range(L):
#                 # ego 车辆无需过滤
#                 if i==0 :
#                     batch_feature.append(x[b][i])
                
#                 # 没有协同车辆
#                 elif mask[b][i]==0 :
#                     batch_feature.append(torch.zeros([C,H,W]).to(x.device))
        
#                 else:
#                     cav_psm=psm[b][i]
#                     single_psm=torch.cat((ego_psm,cav_psm),dim=0).unsqueeze(0)
                    
#                     filter=single_psm.clone()
#                     for conv in self.conv_layers:
#                         filter = conv(filter)
#                     #将l,c,h,w变成 c,h,w ,后面再按车辆叠加
#                     filter=filter[0]

#                     k_percent = self.threshold  # 保留
#                     # 计算保留元素的数量
#                     num_elements = filter.numel()  # 张量中的总元素数
#                     k = max(1, int(num_elements * k_percent))  # 确保至少保留 1 个元素
#                     # 获取前 k 个最大的元素
#                     threshold_value = torch.topk(filter.view(-1), k, sorted=False).values.min()
#                     # 构造掩码
#                     filter_mask = (filter >= threshold_value).int()
#                     batch_feature.append(x[b][i]*filter_mask)
                    
#             filter_feature_list.append(torch.stack(batch_feature,dim=0))
            
#         filter_feature=torch.stack(filter_feature_list,dim=0)
                    

#         return filter_feature





