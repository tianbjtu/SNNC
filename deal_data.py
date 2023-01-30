import torch
import numpy as np
import random

def get_k_fold_data(k,i,x,y):
    assert k>1
    fold_size=x.shape[0]//k
    x_train,y_train=None,None
    for j in range(k):
        idx=slice(j*fold_size,(j+1)*fold_size)
        x_part,y_part=x[idx,:],y[idx]
        if j==i:
            x_valid,y_valid=x_part,y_part
        elif x_train is None:
            x_train,y_train=x_part,y_part
        else:
            x_train=torch.cat((x_train,x_part),dim=0)
            y_train=torch.cat((y_train,y_part),dim=0)
    return x_train,y_train,x_valid,y_valid

def SampleParis(label,paris):
    paris_list=[0]*paris
    for i in range(paris):
        paris_list[i]=[]
    for j in range(paris):
        sample_len=len(label)
        a=random.randint(0,sample_len-1)
        b=random.randint(0,sample_len-1)
        temp=[a,b]
        while(temp in paris_list):
            a=random.randint(0,sample_len-1)
            b=random.randint(0,sample_len-1)
            temp=[a,b]
        paris_list[j].append(a)
        paris_list[j].append(b)
    return paris_list

# def SampleParis(label,paris):    #0  36
#     all=len(label)
#     p=-1
#     paris_list=[0]*paris
#     for dx in range(paris):
#         paris_list[dx]=[]
#     for i in range(all):
#         for j in range(i+1,all):
#             p=p+1
#             paris_list[p].append(i)
#             paris_list[p].append(j)
#     return  paris_list

class SiameseNetworkDataset2():
    def __init__(self, imageFolderDataset, imageFolderDataset2, sample, sample_paris_list):
        self.imageFolderDataset = imageFolderDataset
        self.imageFolderDataset2 = imageFolderDataset2
        self.sample = sample
        self.sample_paris_list = sample_paris_list

    def __getitem__(self, index):
        x1 = self.sample_paris_list[index][0]
        x2 = self.sample_paris_list[index][1]
        p1 = self.imageFolderDataset[x1]
        l1 = self.imageFolderDataset2[x1]
        p2 = self.imageFolderDataset[x2]
        l2 = self.imageFolderDataset2[x2]
        return p1, p2, l1, l2

    def __len__(self):
        return (self.sample)