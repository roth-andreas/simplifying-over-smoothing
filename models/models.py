import torch
from torch_geometric.utils import degree
import torch.nn as nn
import torch_geometric as pyg

from models import ggcn


def dirichlet_energy(x, edge_index, rw=False):
    edge_index = pyg.utils.add_self_loops(edge_index, num_nodes=x.shape[0])[0]
    with torch.no_grad():
        src, dst = edge_index
        deg = degree(src, num_nodes=x.shape[0])
        x = x / torch.norm(x)
        if rw:
            energy = torch.norm(x[src] - x[dst], dim=1, p=2) ** 2.0
        else:
            x = x / torch.sqrt(deg + 0.0).view(-1, 1)
            energy = 1/(torch.sqrt(deg[src])*torch.sqrt(deg[dst])) * torch.norm(x[src] - x[dst], dim=1, p=2) ** 2.0

        #energy = energy.mean()

        energy *= 0.5

    return float(energy.sum().detach().cpu())#.mean()


def rank_diff(x):
    with torch.no_grad():
        x = x / torch.linalg.norm(x, 'nuc')

        i = x.abs().sum(dim=1).argmax()
        j = x.abs().sum(dim=0).argmax()
        mean0 = x[i].view(1, -1)
        mean1 = x[:, j].view(-1, 1)
        if mean0[0,j] < 0:
            mean0 = -mean0
        x_hat = mean1 @ mean0
        x_hat = x_hat / torch.linalg.norm(x_hat, 'nuc')

        return torch.linalg.norm(x - x_hat, 'nuc').item()


class SimpleModel(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, num_layers, conv):
        super().__init__()
        self.enc = nn.Linear(in_dim, h_dim)
        self.conv_type = conv
        self.res = False
        self.norm = False
        self.scaling_factor = 1
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            dim2 = h_dim if i < num_layers - 1 else out_dim
            if conv == 'GCN':
                layer = pyg.nn.GCNConv(h_dim, dim2, bias=True, add_self_loops=True)
            elif conv == 'SAGE':
                layer = pyg.nn.SAGEConv(h_dim, dim2, bias=True)
            elif conv == 'GIN (2 layers)':
                layer = pyg.nn.GINConv(
                    nn.Sequential(nn.Linear(h_dim, dim2, bias=True),
                                  nn.ReLU(),
                                  nn.Linear(dim2, dim2, bias=True),
                                  )
                )
            elif conv == 'GIN (3 layers)':
                layer = pyg.nn.GINConv(
                    nn.Sequential(nn.Linear(h_dim, dim2, bias=True),
                                  nn.ReLU(),
                                  nn.Linear(dim2, dim2, bias=True),
                                  nn.ReLU(),
                                  nn.Linear(dim2, dim2, bias=True),
                                  )
                )
            elif conv == 'GAT':
                layer = pyg.nn.GATv2Conv(h_dim, dim2, heads=2, concat=False)
            elif conv == 'GCNII':
                layer = pyg.nn.GCN2Conv(h_dim, alpha=0.1)
            elif conv == 'GCNII (2x)':
                layer = pyg.nn.GCN2Conv(h_dim, alpha=0.1)
                self.scaling_factor = 2
            elif conv == 'UniMP':
                layer = pyg.nn.TransformerConv(h_dim, dim2, heads=2, concat=False)
            elif conv == 'FAGCN':
                layer = pyg.nn.FAConv(h_dim, eps=0.5)
            elif conv == 'GPS':
                layer = pyg.nn.GPSConv(h_dim, conv=pyg.nn.GCNConv(h_dim, dim2, bias=False, add_self_loops=False),
                                       heads=1)
            elif conv == 'ResGCN':
                layer = pyg.nn.GCNConv(h_dim, dim2, bias=True, add_self_loops=True)
                self.res = True
            elif conv == 'GatedGNN':
                layer = pyg.nn.GatedGraphConv(h_dim, num_layers=1, bias=True)
            elif conv == 'GCN+BatchNorm':
                layer = pyg.nn.GCNConv(h_dim, dim2, bias=True, add_self_loops=True)
                self.norm = True
                self.norms.append(pyg.nn.BatchNorm(dim2))
            elif conv == 'GCN+PairNorm':
                layer = pyg.nn.GCNConv(h_dim, dim2, bias=True, add_self_loops=True)
                # layer = pyg.nn.GCNConv(h_dim, dim2, bias=True, add_self_loops=True)
                self.norm = True
                self.norms.append(pyg.nn.PairNorm(dim2))
            elif conv == 'PPRGNN':
                layer = pyg.nn.GCNConv(h_dim, dim2, bias=True, add_self_loops=True)
            elif conv == 'GGCN':
                layer = ggcn.GGCNlayer(h_dim, dim2, use_degree=True)

            self.convs.append(layer)

        self.num_layers = num_layers

    def forward(self, data):
        x_0 = data.x.float().requires_grad_()
        edge_index = data.edge_index

        energies = []
        energies_sym = []
        rank = []

        x_0 = self.enc(x_0)
        x = x_0
        energies.append(dirichlet_energy(x, edge_index, rw=self.conv_type in ['SAGE', 'DA-SAGE']))
        for i in range(self.num_layers):
            if self.conv_type in ['FAGCN', 'GCNII', 'GCNII (2x)']:
                x_ = self.convs[i](x, x_0, edge_index)
            elif self.conv_type == 'PPRGNN':
                alpha = 1 / (1 + (self.num_layers - i))
                x_ = self.convs[i](x, edge_index)
                x_ = alpha * x_ + x_0
            elif self.conv_type == 'GGCN':
                x_ = self.convs[i](x, pyg.utils.to_dense_adj(edge_index)[0], pyg.utils.degree(edge_index[1], x.size(0)))
            else:
                x_ = self.convs[i](x, edge_index)
            x_ = self.scaling_factor * x_
            if self.norm:
                x_ = self.norms[i](x_)
            if self.res:
                x = x_ + x
            else:
                x = x_

            x = torch.nn.functional.relu(x)
            energies.append(dirichlet_energy(x, edge_index, rw=True))
            energies_sym.append(dirichlet_energy(x, edge_index, rw=False))
            rank.append(rank_diff(x))

        return x, energies, energies_sym, rank
