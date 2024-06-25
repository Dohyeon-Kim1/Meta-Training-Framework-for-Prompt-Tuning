import argparse
import random
import numpy as np
import torch

from model import load_clip_to_cpu, ProtoAttnCoOp, CoOp
from dataset import MetaDTD, DTD
from train import train, train_coop
from test import test


def parse_args():
    parser = argparse.ArgumentParser("Proto Cross Attnetion CoOp")

    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--type', default="base-to-base", type=str)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--n_spt', default=4, type=int)
    parser.add_argument('--n_qry', default=12, type=int)
    parser.add_argument('--n_shot', default=16, type=int)
    parser.add_argument('--model_name', default="model", type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--ckpt', default=None, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if args.mode == "train":
        dataset = MetaDTD(args)
        clip = load_clip_to_cpu()
        model = ProtoAttnCoOp(args, dataset.class_names, clip)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
        train(model, args, dataset, optimizer, lr_sheduler)

    elif args.mode == "test":
        base_dataset = DTD(args, mode="test", domain="base-to-base")
        new_dataset = DTD(args, mode="test", domain="base-to-new") 
        clip = load_clip_to_cpu()

        b_accs, n_accs, h_accs = [], [], []
        for i in range(10, args.epoch+1, 10):
            sd = torch.load(f"checkpoint/{args.ckpt}/{args.seed}/model{i}.pth", map_location="cpu")
            del sd["prompt_learner.token_prefix"], sd["prompt_learner.token_suffix"]

            model = ProtoAttnCoOp(args, base_dataset.class_names, clip)
            model.load_state_dict(sd, strict=False)
            base_acc = test(model, base_dataset) * 100
            
            model = ProtoAttnCoOp(args, new_dataset.class_names, clip)
            model.load_state_dict(sd, strict=False)
            new_acc = test(model, new_dataset) * 100 

            harmonic_acc = (2*base_acc*new_acc) / (base_acc+new_acc)

            b_accs.append(base_acc)
            n_accs.append(new_acc)
            h_accs.append(harmonic_acc)

            print(f"{i}/{args.epoch}: base-to-base: {round(base_acc,2)}%  base-to-new: {round(new_acc,2)}%  harmonic: {round(harmonic_acc,2)}%")

        top_idxs = np.argsort(np.array(h_accs))[-3:]
        print(f"top1({(top_idxs[2]+1)*10}): base-to-base: {round(b_accs[top_idxs[2]],2)}%  base-to-new: {round(n_accs[top_idxs[2]],2)}%  harmonic: {round(h_accs[top_idxs[2]],2)}%")
        print(f"top2({(top_idxs[1]+1)*10}): base-to-base: {round(b_accs[top_idxs[1]],2)}%  base-to-new: {round(n_accs[top_idxs[1]],2)}%  harmonic: {round(h_accs[top_idxs[1]],2)}%")
        print(f"top3({(top_idxs[0]+1)*10}): base-to-base: {round(b_accs[top_idxs[0]],2)}%  base-to-new: {round(n_accs[top_idxs[0]],2)}%  harmonic: {round(h_accs[top_idxs[0]],2)}%")