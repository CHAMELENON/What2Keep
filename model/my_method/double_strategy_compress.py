import torch
import torch.nn as nn
from opencood.tools.visualize_heatmap import feature_heatmap,visualize_batch_heatmaps


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
        # C, H, W = ego_psm.shape

        # residual_psm=ego_psm-cav_psm
        
        # F_updated_flat=self.w1*ego_psm + self.w2*cav_psm
        # F_updated_flat,_=F_updated_flat.sigmoid().max(dim=0, keepdim=True)
        
        
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
        #     aggrate_heatmap(filter_aggrate)


        return filter_mask,filter_residual,filter_aggrate
        
        


class psm_query(nn.Module):
    def __init__(self, args):
        super(psm_query, self).__init__()

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