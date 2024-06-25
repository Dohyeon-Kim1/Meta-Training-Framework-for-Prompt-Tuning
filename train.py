import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model, args, dataset, optim, lr_scheduler=None):
    if not os.path.exists(f"checkpoint/{args.model_name}/{str(args.seed)}/"):
        os.makedirs(f"checkpoint/{args.model_name}/{str(args.seed)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(args.epoch):
        epoch_loss, epoch_acc = [], []
        pbar = tqdm(range(10))
        
        for _ in pbar:
            iter_loss, iter_acc = [], []
            
            for _ in range(args.batch_size):
                spt, qry, class_name = dataset[0]
                spt, qry = spt.to(device), qry.to(device)
                adjusted_feat, qry_feat, text_feat = model(spt, qry, class_name)
                loss = model.get_meta_loss(adjusted_feat, qry_feat, text_feat)
                acc = model.get_meta_acc(adjusted_feat, qry_feat)
                
                iter_loss.append(loss)
                iter_acc.append(acc)

            iter_loss = torch.stack(iter_loss).mean()
            iter_acc = torch.stack(iter_acc).mean()

            epoch_loss.append(iter_loss.detach())
            epoch_acc.append(iter_acc.detach())

            optim.zero_grad()
            iter_loss.backward()
            optim.step()

            pbar.set_postfix(loss=iter_loss.item(), acc=iter_acc.item())

        if lr_scheduler:
            lr_scheduler.step()

        epoch_loss = torch.stack(epoch_loss).mean().item()
        epoch_acc = torch.stack(epoch_acc).mean().item()
        print(f"Epoch {epoch+1}/{args.epoch}  train loss: {epoch_loss}  train acc: {epoch_acc}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint/{args.model_name}/{str(args.seed)}/model{epoch+1}.pth")


def train_coop(model, args, dataset, optim, lr_scheduler=None):
    if not os.path.exists(f"checkpoint/{args.model_name}/{str(args.seed)}/"):
        os.makedirs(f"checkpoint/{args.model_name}/{str(args.seed)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

    model.train()
    for epoch in range(args.epoch):
        epoch_loss, epoch_acc = [], []
        pbar = tqdm(dataloader)

        for data, label in pbar:
            data, label = data.to(device), label.to(device)
            img_feat, text_feat = model(data)

            loss = model.get_loss(img_feat, text_feat, label)
            acc = model.get_acc(img_feat, text_feat, label)

            epoch_loss.append(loss)
            epoch_acc.append(acc)

            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_postfix(loss=loss.item(), acc=acc.item())

        if lr_scheduler:
            lr_scheduler.step()

        epoch_loss = torch.stack(epoch_loss).mean().item()
        epoch_acc = torch.stack(epoch_acc).mean().item()
        print(f"Epoch {epoch+1}/{args.epoch}  train loss: {epoch_loss}  train acc: {epoch_acc}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint/{args.model_name}/{str(args.seed)}/model{epoch+1}.pth")