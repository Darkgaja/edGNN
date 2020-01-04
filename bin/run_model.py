#!/usr/bin/env python3
"""
Run model script.
"""
import torch
import argparse
import importlib

from dgl.data import register_data_args

from utils.inits import to_cuda
from utils.io import print_graph_stats, read_params, create_default_path, remove_model

from core.data.constants import GRAPH, N_RELS, N_CLASSES, N_ENTITIES
from core.models.constants import NODE_CLASSIFICATION, GRAPH_CLASSIFICATION
from core.models.constants import AIFB, MUTAG, MUTAGENICITY, PTC_FM, PTC_FR, PTC_MM, PTC_MR, DPD
from core.models.model import Model
from core.app import App


MODULE = 'core.data.{}'
AVAILABLE_DATASETS = {
    'dglrgcn',
    'dortmund'
}


def main(args):

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)

    

    if args.dataset == AIFB or args.dataset == MUTAG:
        module = importlib.import_module(MODULE.format('dglrgcn'))
        data = module.load_dglrgcn(args.data_path)
        data = to_cuda(data) if cuda else data
        mode = NODE_CLASSIFICATION
    elif args.dataset == MUTAGENICITY or args.dataset == PTC_MR or args.dataset == PTC_MM or args.dataset == PTC_FR or args.dataset == PTC_FM or args.dataset == DPD:
        module = importlib.import_module(MODULE.format('dortmund'))
        data = module.load_dortmund(args.data_path)
        data = to_cuda(data) if cuda else data
        mode = GRAPH_CLASSIFICATION
    else:
        raise ValueError('Unable to load dataset', args.dataset)

    print_graph_stats(data[GRAPH])

    default_path = args.data_path + "model.gnn"
    print('\n*** Set default saving/loading path to:', default_path)

    config_params = read_params(args.config_fpath, verbose=True)

    # create GNN model
    model = Model(g=data[GRAPH],
                  config_params=config_params[0],
                  n_classes=data[N_CLASSES],
                  n_rels=data[N_RELS] if N_RELS in data else None,
                  n_entities=data[N_ENTITIES] if N_ENTITIES in data else None,
                  is_cuda=cuda,
                  mode=mode)

    if cuda:
        model.cuda()

    # 1. Training
    app = App()
    app.model = model
    learning_config = {'lr': args.lr, 'n_epochs': args.n_epochs, 'weight_decay': args.weight_decay, 'batch_size': args.batch_size, 'cuda': cuda}
    if (args.type == "train"):
        print('\n*** Start training ***\n')
        app.train(data, config_params[0], learning_config, default_path, mode=mode)

    if (args.type == "validate"):
        app.validate(data, default_path, mode=mode)
    else:
        app.test(data, default_path, mode=mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run graph neural networks.')
    register_data_args(parser)
    parser.add_argument("--config_fpath", type=str, required=True, 
                        help="Path to JSON configuration file.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path from where to load the data (assuming they were preprocessed beforehand).")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size (only for graph classification)")
    parser.add_argument("--type", type=str, required=True, help="train or validation")

    args = parser.parse_args()

    main(args)
