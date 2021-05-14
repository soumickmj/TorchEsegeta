import torch


class NetCustom(torch.nn.Module):
    def __init__(self, model):
        super(NetCustom, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)[0]
        indx = torch.gt(x, 0.5).float()
        indx1 = torch.le(x, 0.5).float()
        # out = torch.tensor([[torch.sum(model_out), torch.sum(1.0 - model_out)]]).to(self.device)
        # out =  torch.unique(model_out, return_counts = True)[1].unsqueeze(0).float()
        out = ((x / x) * indx).sum(dim=(2, 3, 4))
        out1 = ((x / x) * indx1).sum(dim=(2, 3, 4))
        return torch.cat((out1, out), dim=1)