import torch
import numpy as np
import math


def cal_size(x):
    nonzero_count = torch.count_nonzero(x).item()
    
    # 计算总字节数（每个非零元素占4B）
    total_bytes = nonzero_count * 4
    if total_bytes!=0:
        result=math.log2(total_bytes)
    else:
        result=None
    return result
