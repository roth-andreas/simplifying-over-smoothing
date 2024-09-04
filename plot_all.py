import os

import torch
import torch_geometric as pyg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from models.models import SimpleModel


def get_data(dataset='KarateClub'):
    if dataset == 'Cora':
        dataset = pyg.datasets.Planetoid(root='data/', name='Cora')
        data = dataset[0]
    elif dataset == 'KarateClub':
        data = pyg.datasets.KarateClub()[0]
    data.x = torch.randn((data.x.size(0), 10))
    return data


def run(conv, num_layers):
    data = get_data()

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel(data.num_node_features, 16, 16, num_layers, conv).to(device)

    data.to(device)
    _, energies, energies_sym, rank = model(data)

    return torch.FloatTensor(energies).cpu(), torch.FloatTensor(energies_sym).cpu(), torch.FloatTensor(rank).cpu()


def plot_energies(stat_list, log_y=True, xlabel='Number of layers', ylabel='Rank-one Distance', name='', recolor='all'):
    plt.figure(figsize=(10, 4))
    fontsize = 12
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['text.usetex'] = True
    cm = plt.get_cmap('tab20')
    for i, (conv, stat_list) in enumerate(stat_list.items()):
        if recolor == 'all':
            if i < 8:
                color = i
            else:
                color = 8
        elif recolor == 'constant':
            color = i if i in [3,4,5] else 8
        elif recolor == 'symmetric':
            color = i if i in [0, 1, 2] else 8
        mean = stat_list['mean']
        y = np.arange(0, len(mean))
        linestyle = '-'
        plt.plot(y, mean, linestyle, label=f"{conv}", c=cm(color))
        plus = stat_list['min']
        minus = stat_list['max']
        #plt.fill_between(y, minus, plus, alpha=0.1, color=cm(color))
    if log_y:
        plt.yscale('log')
    plt.legend(ncol=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not os.path.isdir('./figures'):
        os.makedirs('./figures')
    plt.savefig(f'./figures/{name}.svg', bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':

    num_layers = 96
    for eval_set in ['all','constant','symmetric']:
        energy_list = {}
        rank_list = {}
        energy_sym_list = {}
        if eval_set == 'all':
            conv_set = ['GCN', 'GCNII (2x)', 'ResGCN', 'GAT', 'SAGE', 'UniMP', 'GIN (2 layers)', 'GGCN',
                        'GIN (3 layers)',
                        'GPS', 'PPRGNN', 'GCNII', 'GatedGNN', 'GCN+BatchNorm', 'GCN+PairNorm', 'ResGCN']
        elif eval_set == 'constant':
            conv_set = ['GCN', 'GCNII (2x)', 'ResGCN', 'GAT', 'SAGE', 'UniMP', 'GIN (2 layers)', 'GGCN',
                        'GIN (3 layers)',
                        'GPS', 'PPRGNN', 'GCNII', 'GatedGNN', 'GCN+BatchNorm', 'GCN+PairNorm', 'ResGCN']
        elif eval_set == 'symmetric':
            conv_set = ['GCN', 'GCNII (2x)', 'ResGCN', 'GAT', 'SAGE', 'UniMP', 'GIN (2 layers)', 'GGCN',
                        'GIN (3 layers)',
                        'GPS', 'PPRGNN', 'GCNII', 'GatedGNN', 'GCN+BatchNorm', 'GCN+PairNorm', 'ResGCN']
        for conv in conv_set:
            energy_stats = []
            rank_stats = []
            energy_sym_stats = []
            for seed in range(50):
                pyg.seed_everything(seed)
                energy, energy_sym, rank = run(conv, num_layers)
                energy_stats.append(energy)
                rank_stats.append(rank)
                energy_sym_stats.append(energy_sym)
            stats = torch.stack(energy_stats)
            rank = torch.stack(rank_stats)
            energy_sym = torch.stack(energy_sym_stats)
            energy_list[conv] = {'mean': stats.nanmean(0),
                                 'min': torch.nan_to_num(stats, nan=10).min(0)[0],
                                 'max': torch.nan_to_num(stats, nan=1e-10).max(0)[0]}
            rank_list[conv] = {'mean': rank.nanmean(0),
                               'min': torch.nan_to_num(rank, nan=10).min(0)[0],
                               'max': torch.nan_to_num(rank, nan=1e-20).max(0)[0]}
            energy_sym_list[conv] = {'mean': energy_sym.nanmean(0),
                                     'min': torch.nan_to_num(energy_sym, nan=10).min(0)[0],
                                     'max': torch.nan_to_num(energy_sym, nan=1e-20).max(0)[0]}
        if eval_set == 'all':
            plot_energies(rank_list, ylabel='Rank-one distance', name=f'Rank_one_distance_{eval_set}', recolor=eval_set)
        elif eval_set == 'constant':
            plot_energies(energy_list, ylabel='Dirichlet energy $(\mathbf{\Delta} = \mathbf{D} - \mathbf{A})$', name=f'Dirichlet_energy_{eval_set}', recolor=eval_set)
        elif eval_set == 'symmetric':
            plot_energies(energy_sym_list,
                      ylabel=r'Dirichlet energy $\left(\mathbf{\Delta} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\right)$',
                      name=f'dirichlet_energy_symmetric_{eval_set}', recolor=eval_set)
