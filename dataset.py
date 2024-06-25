import json
import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


DATASET_PATH = "/mnt/sdc/dhkim/dataset/dtd"


class MetaDTD(Dataset):
    def __init__(self, args):
        self.args = args
        self.data = []
        self.label = []
        self.class_to_label = {}

        self.transform = transforms.Compose([transforms.Resize((224,224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

        self.train_transform = transforms.Compose([transforms.RandomResizedCrop((224,224)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        
        fpath = f"{DATASET_PATH}/split_zhou_DescribableTextures.json"
        with open(fpath, "r") as f:
            obj = json.load(f)

        for img_path, label, name, in obj[args.mode]:
            if args.type == "base-to-base" and label >= 24:
                continue
            elif args.type == "base-to-new" and label < 24:
                continue
            img = Image.open(f"{DATASET_PATH}/images/{img_path}")
            self.data.append(img)
            self.label.append(label)
            if self.class_to_label.get(name) is None:
                self.class_to_label[name] = label
        
        self.class_names = list(self.class_to_label.keys())
        
        self.sampled_data = []
        self.sampled_label = []
        for class_name in self.class_names:
            label = self.class_to_label[class_name]
            idxs = [idx for idx, value in enumerate(self.label) if label == value]
            random.shuffle(idxs)
            for i in range(self.args.n_shot):
                self.sampled_data.append(self.data[idxs[i]])
                self.sampled_label.append(self.label[idxs[i]])

    def __len__(self):
        return 99999
    
    def __getitem__(self, idx):
        choice_class = random.sample(self.class_names, k=self.args.n_way)
        samples = torch.zeros((self.args.n_way, self.args.n_spt + self.args.n_qry, 3, 224, 224))

        for i, class_name in enumerate(choice_class):
            label = self.class_to_label[class_name]
            idxs = [idx for idx, value in enumerate(self.sampled_label) if label == value]
            random.shuffle(idxs)
            for j in range(self.args.n_spt + self.args.n_qry):
                samples[i,j] = self.train_transform(self.sampled_data[idxs[j]])

        return samples[:,:self.args.n_spt], samples[:,self.args.n_spt:self.args.n_spt+self.args.n_qry], choice_class


class DTD(Dataset):
    def __init__(self, args, mode, domain):
        self.args = args
        self.data = []
        self.label = []
        self.class_to_label = {}

        self.transform = transforms.Compose([transforms.Resize((224,224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

        self.train_transform = transforms.Compose([transforms.RandomResizedCrop((224,224)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

        fpath = f"{DATASET_PATH}/split_zhou_DescribableTextures.json"
        with open(fpath, "r") as f:
            obj = json.load(f)

        for img_path, label, name, in obj[args.mode]:
            if domain == "base-to-base" and label >= 24:
                continue
            elif domain == "base-to-new" and label < 24:
                continue
            img = Image.open(f"{DATASET_PATH}/images/{img_path}")
            self.data.append(img)
            self.label.append(label if domain == "base-to-base" else label-24)
            if self.class_to_label.get(name) is None:
                self.class_to_label[name] = label
        
        self.class_names = list(self.class_to_label.keys())

        self.sampled_data = []
        self.sampled_label = []
        if mode == "train":
            for class_name in self.class_names:
                label = self.class_to_label[class_name]
                idxs = [idx for idx, value in enumerate(self.label) if label == value]
                random.shuffle(idxs)
                for i in range(self.args.n_shot):
                    self.sampled_data.append(self.data[idxs[i]])
                    self.sampled_label.append(self.label[idxs[i]])
        else:
            self.sampled_data = self.data
            self.sampled_label = self.label
        
    def __len__(self):
        return len(self.sampled_data)
    
    def __getitem__(self, idx):
        if self.args.mode == "train":
            return self.train_transform(self.sampled_data[idx]), self.sampled_label[idx]
        else:
            return self.transform(self.sampled_data[idx]), self.sampled_label[idx]
