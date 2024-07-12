import torch
import torch.nn as nn
import torch.nn.functional as F


class GGCNlayer(nn.Module):
    def __init__(self, in_features, out_features, use_degree=True, use_sign=True, use_decay=True, scale_init=0.5,
                 deg_intercept_init=0.5):
        super(GGCNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fcn = nn.Linear(in_features, out_features)
        self.use_degree = use_degree
        self.use_sign = use_sign
        if use_degree:
            if use_decay:
                self.deg_coeff = nn.Parameter(torch.tensor([0.5, 0.0]))
            else:
                self.deg_coeff = nn.Parameter(torch.tensor([deg_intercept_init, 0.0]))
        if use_sign:
            self.coeff = nn.Parameter(0 * torch.ones([3]))
            if use_decay:
                self.scale = nn.Parameter(2 * torch.ones([1]))
            else:
                self.scale = nn.Parameter(scale_init * torch.ones([1]))
        self.sftmax = nn.Softmax(dim=-1)
        self.sftpls = nn.Softplus(beta=1)

    def forward(self, h, adj, degree_precompute):
        if self.use_degree:
            sc = self.deg_coeff[0] * degree_precompute + self.deg_coeff[1]
            sc = self.sftpls(sc)

        Wh = self.fcn(h)
        if self.use_sign:
            prod = torch.matmul(Wh, torch.transpose(Wh, 0, 1))
            sq = torch.unsqueeze(torch.diag(prod), 1)
            scaling = torch.matmul(sq, torch.transpose(sq, 0, 1))
            e = prod / torch.max(torch.sqrt(scaling), 1e-9 * torch.ones_like(scaling))
            e = e - torch.diag(torch.diag(e))
            if self.use_degree:
                attention = e * adj * sc
            else:
                attention = e * adj

            attention_pos = F.relu(attention)
            attention_neg = -F.relu(-attention)
            prop_pos = torch.matmul(attention_pos, Wh)
            prop_neg = torch.matmul(attention_neg, Wh)

            coeff = self.sftmax(self.coeff)
            scale = self.sftpls(self.scale)
            result = scale * (coeff[0] * prop_pos + coeff[1] * prop_neg + coeff[2] * Wh)

        else:
            if self.use_degree:
                prop = torch.matmul(adj * sc, Wh)
            else:
                prop = torch.matmul(adj, Wh)

            result = prop

        return result
