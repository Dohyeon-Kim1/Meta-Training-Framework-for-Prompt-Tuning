import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def meta_test(model, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.eval()
    iter_acc = []
    for i in range(10):
        spt, qry, class_name = dataset[0]
        spt, qry = spt.to(device), qry.to(device)
        adjusted_feat, qry_feat, text_feat = model(spt, qry, class_name)
        acc = model.get_acc(text_feat, qry_feat)
                
        iter_acc.append(acc)

    acc = torch.stack(iter_acc).mean().item()
    print(f"Test Accuracy: {acc}")


@torch.no_grad()
def test(model, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, pin_memory=True)

    model.eval()
    acc_list = []
    for data, label in dataloader:
        data, label = data.to(device), label.to(device)
        text_feat = model.get_text_feat()
        img_feat = model.get_image_feat(data)
        acc = model.get_acc(img_feat, text_feat, label)

        acc_list.append(acc)
    
    acc = torch.stack(acc_list).mean().item()
    return acc