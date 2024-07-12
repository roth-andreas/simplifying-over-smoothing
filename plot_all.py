import os

import torch
import torch_geometric as pyg
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleModel


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
    _, energies, rank = model(data)

    return torch.FloatTensor(energies).cpu(), torch.FloatTensor(rank).cpu()


def plot_energies(stat_list, log_y=True, xlabel='Number of layers', ylabel='Rank-one Distance', postfix=''):
    plt.figure(figsize=(10, 4))
    fontsize = 12
    plt.rcParams.update({'font.size': fontsize})
    cm = plt.get_cmap('tab20')
    for i, (conv, stat_list) in enumerate(stat_list.items()):
        mean = stat_list['mean']
        y = np.arange(0, len(mean))
        linestyle = '-'
        plt.plot(y, mean, linestyle, label=f"{conv}", c=cm(i))
        plus = stat_list['min']
        minus = stat_list['max']
        plt.fill_between(y, minus, plus, alpha=0.1, color=cm(i))
    if log_y:
        plt.yscale('log')
    plt.legend(ncol=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not os.path.isdir('./figures'):
        os.makedirs('./figures')
    plt.savefig(f'./figures/{ylabel}_{postfix}.svg', bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':

    num_layers = 96
    energy_list = {}
    rank_list = {}
    for eval_set in ['all', 'collapse', 'rest']:
        if eval_set == 'all':
            conv_set = ['GCN', 'GAT', 'SAGE', 'UniMP', 'GIN (2 layers)', 'GGCN', 'GCNII (2x)', 'ResGCN',
                        'GIN (3 layers)',
                        'GPS', 'PPRGNN', 'GCNII', 'GatedGNN', 'GCN+BatchNorm', 'GCN+PairNorm', 'ResGCN']
        elif eval_set == 'collapse':
            conv_set = ['GCN', 'GAT', 'SAGE', 'UniMP', 'GIN (2 layers)', 'GGCN', 'GCNII (2x)', 'ResGCN']
        else:
            conv_set = ['GIN (3 layers)', 'GPS', 'PPRGNN', 'GCNII', 'GatedGNN', 'GCN+BatchNorm', 'GCN+PairNorm',
                        'ResGCN']
        for conv in conv_set:
            conv_stats = []
            rank_stats = []
            for seed in range(50):
                pyg.seed_everything(seed)
                dir_en, rank = run(conv, num_layers)
                conv_stats.append(dir_en)
                rank_stats.append(rank)
            stats = torch.stack(conv_stats)
            rank = torch.stack(rank_stats)
            energy_list[conv] = {'mean': stats.nanmean(0),
                                 'min': torch.nan_to_num(stats, nan=10).min(0)[0],
                                 'max': torch.nan_to_num(stats, nan=1e-10).max(0)[0]}
            rank_list[conv] = {'mean': rank.nanmean(0),
                               'min': torch.nan_to_num(rank, nan=10).min(0)[0],
                               'max': torch.nan_to_num(rank, nan=1e-20).max(0)[0]}
        plot_energies(rank_list, ylabel='Rank-one distance', postfix=eval_set)
        plot_energies(energy_list, ylabel='Dirichlet energy', postfix=eval_set)
