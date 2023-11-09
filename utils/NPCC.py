import torch
from torch import mean, sqrt

# 函数形式，方便计算重建图像的负皮尔森系数
def PCC_function(prediction, label):
    _, _, h, w = prediction.size()
    prediction = prediction.view(h * w, -1)
    label = label.view(h * w, -1)

    mean_prediction = mean(prediction, dim=0, keepdim=True)
    mean_label = mean(label, dim=0, keepdim=True)

    d_prediction = prediction - mean_prediction
    d_label = label - mean_label
    PCC = mean(d_prediction * d_label, dim=0) / (sqrt(mean(d_prediction**2, dim=0)+1e-08) * sqrt(mean(d_label**2, dim=0)+1e-08))
    
    PCC = torch.mean(PCC)

    return PCC