import argparse
import json
import pickle
from collections import defaultdict, Counter
from os.path import dirname, join
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
from base_model import GenB, Discriminator

from train import train
import utils
import click


def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument('--cache_features', default=False, help="Cache image features in RAM. Makes things much faster"
                        "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument('--dataset', default='cpv2', choices=["v2", "cpv2", "cpv1"], help="Run on VQA-2.0 instead of VQA-CP 2.0")
    parser.add_argument('--eval_each_epoch', default=True,help="Evaluate every epoch, instead of at the end")
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='logs/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--load_checkpoint_path', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset=args.dataset
    args.output=os.path.join('logs',args.output)
    if not os.path.isdir(args.output):
        utils.create_dir(args.output)
    else:
        if click.confirm('Exp directory already exists in {}. Erase?'
                                 .format(args.output, default=False)):
            os.system('rm -r ' + args.output)
            utils.create_dir(args.output)

        else:
            if args.load_checkpoint_path is None:
                os._exit(1)


    if dataset=='cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    elif dataset=='cpv2' or dataset=='v2':
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building train dataset...")
    train_dset = VQAFeatureDataset('train', dictionary, dataset=dataset,
                                #    cache_image_features=args.cache_features)
                                cache_image_features=False)

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                #   cache_image_features=args.cache_features)
                                cache_image_features=False)

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    genb = GenB(num_hid=1024, dataset=train_dset).cuda()
    discriminator = Discriminator(num_hid=1024, dataset=train_dset).cuda()
    if dataset=='cpv1':
        model.w_emb.init_embedding('data/glove6b_init_300d_v1.npy')
        genb.w_emb.init_embedding('data/glove6b_init_300d_v1.npy')
    elif dataset=='cpv2' or dataset=='v2':
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')
        genb.w_emb.init_embedding('data/glove6b_init_300d.npy')

    with open('util/qid2type_%s.json'%args.dataset,'r') as f:
        qid2type=json.load(f)

    if args.load_checkpoint_path is not None:
        ckpt = torch.load(os.path.join('logs', args.load_checkpoint_path, 'model.pth'))
        model_dict = model.state_dict()
        ckpt = {k: v for k, v in ckpt.items() if k in model_dict}
        model_dict.update(model_dict)
        model.load_state_dict(model_dict)

    model=model.cuda()
    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)

    print("Starting training...")
    train(model, genb, discriminator, train_loader, eval_loader, args,qid2type)

if __name__ == '__main__':
    main()
