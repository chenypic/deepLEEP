import os
import glob
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import natsort
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from sklearn import preprocessing

means = [0.5041255852915447, 0.5039916594753655, 0.503902651427799]
stds = [0.18183103394930122, 0.18186770685108816, 0.18191354305832244]

means = [0.5, 0.5, 0.5]
stds = [0.5, 0.5, 0.5]

normalize = transforms.Normalize(
    mean=means,
    std=stds,
)

train_transform = transforms.Compose([
    # 4 pixels are padded on each side, 
    # transforms.Pad(4),
    # a 32×32 crop is randomly sampled from the 
    # padded image or its horizontal flip.
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomCrop(32),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    # For testing, we only evaluate the single 
    # view of the original 32×32 image.
    transforms.ToTensor(),
    normalize
])

def read_and_resize(file):
    img = cv2.imread(file)
    # img_rgb2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb2 = img[:,:,::-1].copy()
    ## resized = cv2.resize(img_rgb2, (256, 256), interpolation=cv2.INTER_LINEAR)
    # croppeds = seq(image=resized).copy()

    #return cropped
    return img_rgb2

def read_and_resize_val(file):
    img = cv2.imread(file)
    # img_rgb2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb2 = img[:,:,::-1].copy()
    # resized = cv2.resize(img_rgb2, (256, 256), interpolation=cv2.INTER_LINEAR)

    return img_rgb2


class Genedataset(Dataset):
    """Gene dataset."""

    def __init__(self, data_dir,pd_data, transform=None,is_train = True):
        """
        Args:
            data_file (string): 
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train

        self.imgs = glob.glob(os.path.join(self.data_dir,'*/*.jpg'))
        self.cli = pd.read_excel(pd_data,index_col=0)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        p_id = int(os.path.basename(self.imgs[idx]).split('_')[1])

        p_cli = list(self.cli.iloc[p_id][['TCT》ASCUS', 'TCT》HSIL', 'HPV16/18', '绝经', '孕次》4','产次》3', '转化区类型', '镜下评估']])
        p_cli = list(map(int, p_cli))

        ori_dir = self.cli.iloc[p_id]['dir']
        p_name = self.cli.iloc[p_id]['姓名']

        if self.is_train:
            img = read_and_resize(self.imgs[idx])
        else:
            img = read_and_resize_val(self.imgs[idx])
        # img = img.transpose([2, 0, 1])
        
        label = os.path.basename(os.path.dirname(self.imgs[idx]))

        if label == 'positive':
            label = 1
        else:
            label = 0

        sample = {'img': img, 'label': label,'file':self.imgs[idx],'cli':p_cli,'ori_dir':ori_dir,'p_name':p_name}


        if self.transform:
            sample['img'] = self.transform(sample['img'])

        return sample
    

data_dir = '/data/ichenwei/yzhr/002-qieyuan/data/qilu_clean3/'
pd_data = '齐鲁 LEEP数据 2024.9.13-链接-2。1.xlsx'

trainData = Genedataset(data_dir,pd_data,transform=train_transform)


class Genedataset_Val(Dataset):
    """Gene dataset."""

    def __init__(self,transform=None):
        """
        Args:
            data_file (string): 
        """
        self.data_dir = '/data/ichenwei/yzhr/002-qieyuan/data/eryuan_clean'
        self.transform = transform

        self.imgs = glob.glob(os.path.join(self.data_dir,'*/*.jpg'))
        self.cli = pd.read_csv("/data/ichenwei/yzhr/002-qieyuan/code/二院表格/二院all.csv",index_col=0)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        p_id = os.path.basename(self.imgs[idx]).split('ori')[0]

        p_cli = self.cli.loc[self.cli['img_ids']==p_id][['TCT》ASCUS', 'TCT》HSIL', 'HPV', '绝经', '孕次','产次', '转化区', '镜下']].values


  
        img = read_and_resize_val(self.imgs[idx])
      
        
        label = self.cli.loc[self.cli['img_ids']==p_id]['label'].values[0]


        sample = {'img': img, 'label': label,'file':self.imgs[idx],'cli':p_cli,'p_name':p_id}


        if self.transform:
            sample['img'] = self.transform(sample['img'])

        return sample
    

valData = Genedataset_Val(transform=test_transform)
