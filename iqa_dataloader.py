import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, Normalize, ToPILImage
import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import scipy.misc as m
import cv2
from tqdm import tqdm


class KONIDataset(Dataset):
    def __init__(self, csv_path, data_path, img_size, transforms=None, is_train=True):
        super().__init__()
        self.transforms = transforms
        self.is_train = is_train
        self.p = 0.5
        self.img_size = img_size
        self.csv_path = csv_path
        self.data_path = data_path
        self.data_tmp, self.dist_img_path, self.dist_type, self.dmos = self.csv_read(self.csv_path)

        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split(self.data_tmp, self.dmos, test_size=0.2, random_state=2, shuffle=True)

    def __getitem__(self, idx):
        if self.is_train:
            dist_img = self.img_read(self.data_path, self.x_train[idx][0], self.img_size)
            dist_img = self.to_tensor(dist_img)

            return dist_img, self.y_train[idx]
        else:
            dist_img = self.img_read(self.data_path, self.x_test[idx][0], self.img_size)
            dist_img = self.to_tensor(dist_img)
            return dist_img, self.y_test[idx]

    def __len__(self):
        if self.is_train:
            return len(self.y_train)
        else:
            return len(self.y_test)

    def train_test_split(self, X, y, test_size, random_state, shuffle):
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        return x_train, x_test, y_train, y_test

    def csv_read(self, csv_path):
        dist_img_path = []
        dist_type = []
        dmos = []
        data_tmp = []
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            # for row in reader:
            for row in tqdm(reader):
                data_tmp.append(row[0:3])
                dist_img_path.append(row[0])
                dist_type.append(row[1])
                dmos.append(float(row[2]))
        return data_tmp, dist_img_path, dist_type, dmos

    def img_read(self, data_path, dist, img_size):
        dist = dist[dist.rfind('/')+1:]
        # print(data_path + dist, os.path.exists(data_path + dist))
        dist_img = cv2.imread(data_path + dist, cv2.IMREAD_COLOR)
        dist_img = cv2.resize(dist_img, (img_size, img_size))
        dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)

        return dist_img


    def to_tensor(self, dist_img):
        if self.is_train == True:
            if torch.rand(1) < self.p:
                to_pil = ToPILImage()
                dist_img = to_pil(dist_img)
                dist_img = F.hflip(dist_img)

        totensor = ToTensor()
        dist_img = totensor(dist_img)
        F.normalize(dist_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return dist_img



if __name__ == "__main__":
    csv_path = 'C:\\Users\\yunjeongyong\\Desktop\\intern\\Triq-yunjeong\\data\\all_data_csv\\KonIQ-10k.txt.csv'
    data_path = 'C:\\Users\\yunjeongyong\\Desktop\\intern\\Triq-yunjeong\\data\\1024x768'
    dataset = KONIDataset(csv_path, data_path, transforms=None, is_train=True)
    for i in range(10):
        dist_img, ref_img, error_img, dmos = dataset[i]
    print(0)
