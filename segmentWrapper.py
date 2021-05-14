import torch
import torch.nn as nn
import numpy as np
from skimage import filters


class SegmentWrapper(nn.Module):
    def __init__(self, baseModel):
        super(SegmentWrapper, self).__init__()
        self.base = baseModel

    def forward(self, x):
        out = self.base(x)
        if type(out) is list:
            out = out[0]

        op = out.squeeze().cpu().detach().numpy()
        op = (op - np.min(op)) / (np.max(op) - np.min(op))
        val = filters.threshold_otsu(op)
        indx = torch.gt(out, val).float()
        indx1 = torch.le(out, val).float()
        # out = torch.tensor([[torch.sum(model_out), torch.sum(1.0 - model_out)]]).to(self.device)
        # out =  torch.unique(model_out, return_counts = True)[1].unsqueeze(0).float()
        op1 = ((out / out) * indx).sum(dim=(2, 3, 4))
        op2 = ((out / out) * indx1).sum(dim=(2, 3, 4))
        return torch.cat((op2, op1), dim=1)


class MultiClassSegmentWrapper(nn.Module):
    def __init__(self, baseModel):
        super(MultiClassSegmentWrapper, self).__init__()
        self.base = baseModel

    def forward(self, x):
        out = self.base(x)
        if type(out) is list:
            out = out[0]
        op_max = torch.argmax(out, dim=1, keepdim=True)
        selected_inds = torch.zeros_like(out[0:]).scatter_(1, op_max, 1)
        return (out * selected_inds).sum(dim=(-2, -1))


