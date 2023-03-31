import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random
import copy
import time
import torch.nn.functional as F

Tensor = torch.cuda.FloatTensor


def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def calc_genb_loss(logits, bias, labels):
    gen_grad = torch.clamp(2 * labels * torch.sigmoid(-2 * labels * bias.detach()), 0, 1)
    loss = F.binary_cross_entropy_with_logits(logits, gen_grad)
    loss *= labels.size(1)
    return loss


def train(model, genb, discriminator, train_loader, eval_loader,args,qid2type):
    num_epochs=args.epochs
    run_eval=args.eval_each_epoch
    output=args.output
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    optim_G = torch.optim.Adamax(filter(lambda p: p.requires_grad, genb.parameters()), lr=0.001)
    optim_D = torch.optim.Adamax(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=0.001)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0

    genb.train(True)
    discriminator.train(True)

    kld = nn.KLDivLoss(reduction='batchmean')
    bce = nn.BCELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()
        for i, (v, q, a, qid) in tqdm(enumerate(train_loader), ncols=100, desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            total_step += 1

            #########################################
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            valid = Variable(Tensor(v.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(v.size(0), 1).fill_(0.0), requires_grad=False)
            #########################################

            # get model output
            optim.zero_grad()
            pred = model(v, q)

            # train genb
            optim_G.zero_grad()
            optim_D.zero_grad()

            pred_g = genb(v, q, gen=True)
            g_loss = F.binary_cross_entropy_with_logits(pred_g, a, reduction='none').mean()
            g_loss *= a.size(1)

            vae_preds = discriminator(pred_g)
            main_preds = discriminator(pred)

            g_distill = kld(pred_g, pred.detach())
            dsc_loss = bce(vae_preds, valid) + bce(main_preds, valid)

            g_loss += dsc_loss + g_distill*5
            g_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(genb.parameters(), 0.25)
            optim_G.step()
            # done training genb

            # train the discriminator
            dsc_loss = bce(vae_preds, fake) + bce(main_preds, valid)
            dsc_loss.backward(retain_graph=True)
            optim_D.step()
            # done training the discriminator

            # use genb to train the robust model
            genb.train(False)
            pred_g = genb(v, q, gen=False)

            genb_loss = calc_genb_loss(pred, pred_g, a)
            genb_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()

            genb.train(True)
            total_loss += genb_loss.item() * q.size(0)
            batch_score = compute_score_with_logits(pred, a.data).sum()
            train_score += batch_score


        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        logger.write('Epoch %d, time: %.2f' % (epoch + 1, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader, qid2type)
            results["epoch"] = epoch
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))
            main_eval_score = eval_score

            if main_eval_score > best_eval_score:
                model_path = os.path.join(output, 'model.pth')
                genb_path = os.path.join(output, 'genb.pth')
                torch.save(model.state_dict(), model_path)
                torch.save(genb.state_dict(), genb_path)
                best_eval_score = main_eval_score

        model_path = os.path.join(output, 'model_final.pth')
        torch.save(model.state_dict(), model_path)
    print('best eval score: %.2f' % (best_eval_score*100))


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

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )
    return results
