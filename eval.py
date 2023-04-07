import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from dataset import Dictionary, VQAFeatureDataset
import base_model

from torch.autograd import Variable

from tqdm import tqdm



def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def evaluate(model, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0 

    for v, q, a, qids in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        pred = model(v, q)
        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    return score, upper_bound, score_yesno, score_other, score_number


def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument('--cache_features', default=False, help="Cache image features in RAM. Makes things much faster"
                        "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument('--dataset', default='cpv2', choices=["v2", "cpv2", "cpv1"], help="Run on VQA-2.0 instead of VQA-CP 2.0")
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--load_path', type=str, default='best_model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset=args.dataset

    if dataset=='cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    elif dataset=='cpv2' or dataset=='v2':
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                cache_image_features=args.cache_features)

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()

    with open('util/qid2type_%s.json'%args.dataset,'r') as f:
        qid2type=json.load(f)

    ckpt = torch.load(os.path.join(args.load_path, 'model.pth'))
    model.load_state_dict(ckpt)
    print('Loaded Model!')

    model=model.cuda()
    model.eval()
    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)
    eval_score, bound, yn, other, num = evaluate(model, eval_loader, qid2type)
    print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
    print('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))

if __name__ == '__main__':
    main()
