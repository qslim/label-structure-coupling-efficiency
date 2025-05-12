import time
import yaml
import copy
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, r2_score
from utils import count_parameters, init_params, seed_everything, get_split


def main_worker(args, config):
    print(args, config)
    seed_everything(args.seed)
    # device = 'cuda:{}'.format(args.cuda)
    # torch.cuda.set_device(args.seed)

    epoch = config['epoch']
    lr = config['lr']
    weight_decay = config['weight_decay']
    nclass = config['nclass']
    nlayer = config['nlayer']
    hidden_dim = config['hidden_dim']
    num_heads = config['num_heads']
    tran_dropout = config['tran_dropout']
    feat_dropout = config['feat_dropout']
    prop_dropout = config['prop_dropout']
    norm = config['norm']

    if args.dataset in ['arxiv', 'products']:
        e1, u1, _, _ = torch.load('../../ogbn_dataset/' + args.dataset + 'x.pt'.format(args.dataset))
        e2, u2 = torch.load('../../ogbn_dataset/' + args.dataset + 'x.pt'.format(args.dataset))
        e, u = torch.cat((e1, e2), dim=0), torch.cat((u1, u2), dim=1)
        x, y = torch.load('../../ogbn_dataset/' + args.dataset + 'x.pt'.format(args.dataset))
    else:
        e, u, x, y = torch.load('data/{}.pt'.format(args.dataset))

    e, u, x, y = e.cuda(), u.cuda(), x.cuda(), y.cuda()
    if len(y.size()) > 1:
        if y.size(1) > 1:
            y = torch.argmax(y, dim=1)
        else:
            y = y.view(-1)

    train, valid, test = get_split(args.dataset, y, nclass, args.seed)
    train, valid, test = map(torch.LongTensor, (train, valid, test))
    train, valid, test = train.cuda(), valid.cuda(), test.cuda()

    print(e.shape)
    print(u.shape)

    nfeat = x.size(1)
    if args.model == 'label_structure_coupling':
        nonlinear = config['nonlinear']
        patience = config['patience']
        residual = config['residual'] == 'True'
        fp_inter = config['fp_inter'] == 'True'
        if args.dataset in ['arxiv', 'products']:
            from label_structure_coupling import LabelStructureCoupling2
            net = LabelStructureCoupling2(nclass, nfeat, nlayer, hidden_dim, num_heads, tran_dropout, feat_dropout, prop_dropout, nonlinear, residual, fp_inter).cuda()
        else:
            from label_structure_coupling import LabelStructureCoupling
            if args.dataset == 'pubmed':
                layer_nonlinear = False
            else:
                layer_nonlinear = True
            net = LabelStructureCoupling(nclass, nfeat, nlayer, hidden_dim, num_heads, tran_dropout, feat_dropout, prop_dropout, nonlinear, residual, fp_inter, layer_nonlinear=layer_nonlinear).cuda()
    else:
        raise NotImplementedError
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    print(count_parameters(net))

    res = []
    min_loss = 100.0
    max_acc = 0
    counter = 0
    evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)

    cur_best_a, cur_best_b, cur_best_c = 0.0, 0.0, 0.0
    for idx in range(epoch):

        net.train()
        optimizer.zero_grad()
        logits = net(e, u, x)

        loss = F.cross_entropy(logits[train], y[train])

        loss.backward()
        optimizer.step()

        net.eval()
        logits = net(e, u, x)

        val_loss = F.cross_entropy(logits[valid], y[valid]).item()

        val_acc = evaluation(logits[valid].cpu(), y[valid].cpu()).item()
        test_acc = evaluation(logits[test].cpu(), y[test].cpu()).item()
        res.append([val_loss, val_acc, test_acc])

        if test_acc > cur_best_a:
            cur_best_a = test_acc
        elif test_acc > cur_best_b:
            cur_best_b = test_acc
        elif test_acc > cur_best_c:
            cur_best_c = test_acc

        print('{}, {:.8f}, {:.8f}, {:.8f}               {:.8f}, {:.8f}, {:.8f}'.format(idx, val_loss, val_acc, test_acc, cur_best_c, cur_best_b, cur_best_a))

        if val_loss < min_loss:
            min_loss = val_loss
            counter = 0
        else:
            counter += 1

        if counter == patience:
            break

    max_acc1 = sorted(res, key=lambda x: x[0], reverse=False)[0][-1]
    max_acc2 = sorted(res, key=lambda x: x[1], reverse=True)[0][-1]
    return max_acc1, max_acc2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--model', default='label_structure_coupling')

    args = parser.parse_args()

    config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[args.dataset]

    _acc1, _acc2 = [], []
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    config['dataset'] = args.dataset
    config['cuda'] = args.cuda
    config['seeds'] = seeds
    config['model'] = args.model
    config['rank'] = 0

    # import wandb
    # wandb.login()
    for seed in seeds:
        args.seed = seed
        acc1, acc2 = main_worker(args, config)
        acc1, acc2 = acc1 * 100.0, acc2 * 100.0
        print(config)
        print(acc1, acc2)
        _acc1.append(acc1)
        _acc2.append(acc2)

    _acc1, _acc2 = np.array(_acc1, dtype=float), np.array(_acc2, dtype=float)
    ACC1 = "{:.2f}, {:.2f}".format(np.mean(_acc1), np.std(_acc1))
    ACC2 = "{:.2f}, {:.2f}".format(np.mean(_acc2), np.std(_acc2))
    print("Mean over {} run".format(len(seeds)), "Acc1:" + ACC1, "Auc2:" + ACC2)



