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



class single_para(nn.Module):
    def __init__(self, args):
        super(single_para, self).__init__()
        

        self.feature_dim = args['anchor_nums']  # 特征图的通道数
        self.dim=args['dim']
        self.comm_volume=args['comm_volume']

        self.w1 = 0.3
        self.w2 = 0.7
        

    def forward(self,ego_psm,cav_psm):
        # C, H, W = ego_psm.shape

        # residual_psm=ego_psm-cav_psm
        
        # F_updated_flat=self.w1*ego_psm + self.w2*cav_psm
        # F_updated_flat,_=F_updated_flat.sigmoid().max(dim=0, keepdim=True)
        
        
        C, H, W = ego_psm.shape
        residual_psm=cav_psm-ego_psm
        aggrated_psm=cav_psm+ego_psm
        psm=residual_psm
        F_updated_flat,_=psm.sigmoid().max(dim=0, keepdim=True)
        
        
        
        if self.training:
            k_percent= torch.rand(1)
            num_elements = F_updated_flat.numel()  # 张量中的总元素数
            k = max(1, int(num_elements * k_percent))  # 确保至少保留 1 个元素
        else:
            k=min(int(2**self.comm_volume/4/self.dim),F_updated_flat.numel())
    
        # 获取前 k 个最大的元素
        threshold_value = torch.topk(F_updated_flat.view(-1), k, sorted=False).values.min()
        # 构造掩码
        filter_mask = (F_updated_flat >= threshold_value).float()
        
        return filter_mask
        
     
class psm_query(nn.Module):
    def __init__(self, args):
        super(psm_query, self).__init__()
        
        self.filter= single_para(args)

        
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
                    filter_mask=self.filter(ego_psm,cav_psm)
                    batch_feature.append(x[b][i]*filter_mask)
                    
            filter_feature_list.append(torch.stack(batch_feature,dim=0))
            
        filter_feature=torch.stack(filter_feature_list,dim=0)
                    

        return filter_feature


