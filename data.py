# import os
import csv
import math
import torch
import torch.nn.functional as F
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
# from utils import list_pictures
from torchvision import transforms
from torch.utils.data import dataloader
from PIL import Image
import pandas as pd
from pathlib import Path


class Data:
    def __init__(self, args, data_type):
        self.args = args
        # transform_list = [
        #     transforms.RandomChoice(
        #         [transforms.RandomHorizontalFlip(),
        #          transforms.RandomGrayscale(),
        #          transforms.RandomRotation(20),
        #          ]
        #     ),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        #                          0.229, 0.224, 0.225])
        # ]
        # transform = transforms.Compose(transform_list)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        self.train_dataset = Dataset(args, data_type, transform)
        self.train_loader = dataloader.DataLoader(self.train_dataset,
                                                  shuffle=True,
                                                  batch_size=args.train_batch_size,
                                                  num_workers=args.nThread
                                                  )
        self.valid_loader = dataloader.DataLoader(self.train_dataset,
                                                  shuffle=False,
                                                  batch_size=args.train_batch_size,
                                                  num_workers=args.nThread
                                                  )


class Dataset(dataset.Dataset):
    def __init__(self, args, data_type, transform):
        # self.root = args.train_img
        # self.transform = transform
        # self.labels = [label[0:-1]
        #                for label in csv.reader(open(args.train_label, 'r'))]
        self.loader = default_loader
        self.transform = transform

        csv_path = Path(args.data_dir).joinpath(
            f"{args.dataset}_{data_type}_align.csv")
        img_dir = Path(args.data_dir)
        # self.img_size = img_size
        # self.augment = augment
        # self.age_stddev = age_stddev

        # if augment:
        #     self.transform = ImgAugTransform()
        # else:
        #     self.transform = lambda i: i

        self.x = []
        self.y = []
        # self.std = []
        self.rotate = []
        self.boxes = []
        df = pd.read_csv(str(csv_path))

        for _, row in df.iterrows():
            img_name = row["photo"]
            img_path = img_dir.joinpath(img_name)
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["age"])
            self.rotate.append(row["deg"])
            self.boxes.append(
                [row["box1"], row["box2"], row["box3"], row["box4"]])
            # self.std.append(row["apparent_age_std"])

    def __getitem__(self, idx):
        img_path = self.x[idx]
        img = Image.open(img_path)
        if img.mode is not "RGB":
            img = img.convert("RGB")
        # img.show()
        img = img.rotate(
            self.rotate[idx], resample=Image.BICUBIC, expand=True)  # Alignment
        img = img.crop(self.boxes[idx])
        # img.show()
        img = self.transform(img)
        # print(img.shape)

        age = self.y[idx]

        label = [normal_sampling(int(age), i) for i in range(101)]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)
        return img, label, int(age)

    def __len__(self):
        return len(self.y)


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)
