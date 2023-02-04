import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim
from torch.utils.data import DataLoader,Dataset
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
from net import SNNC
from deal_data import get_k_fold_data
from deal_data import SampleParis
from deal_data import SiameseNetworkDataset2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, train_features, test_features, train_label, test_label, train_sample, train_sample_paris_list):
    siamese_dataset_train = SiameseNetworkDataset2(imageFolderDataset=train_features,
                                                   imageFolderDataset2=train_label,
                                                   sample=train_sample,
                                                   sample_paris_list=train_sample_paris_list)

    train_dataloader = DataLoader(dataset=siamese_dataset_train,
                                  shuffle=True,
                                  num_workers=0,
                                  batch_size=64)

    train_ls = []
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    loss = torch.nn.MSELoss()
    loss = loss.to(device)

    # train model
    for epoch in range(100):
        for i, data in enumerate(train_dataloader, 0):
            person1, person2, label1, label2 = data
            person1 = person1.to(device)
            label1 = label1.to(device)
            person2 = person2.to(device)
            label2 = label2.to(device)
            person1 = torch.unsqueeze(person1, 1)
            person2 = torch.unsqueeze(person2, 1)
            person1 = person1.to(torch.float32)
            person2 = person2.to(torch.float32)
            d12 = net(person1, person2)
            y12 = label1 - label2
            d12 = d12.to(torch.float32)
            y12 = y12.to(torch.float32)
            loss_age = loss(d12, y12)
            optimizer.zero_grad()
            loss_age.backward()
            optimizer.step()
            if i == 0:
                train_ls.append(loss_age.item())

    # test model
    with torch.no_grad():
        for j in range(len(test_label)):
            person1 = test_features[j]
            age1 = test_label[j]
            person1 = torch.Tensor(person1)
            person1 = person1.to(device)
            person1 = torch.unsqueeze(person1, 0)
            person1 = torch.unsqueeze(person1, 1)
            person1 = person1.to(torch.float32)
            # output1= net(person1.to(torch.float32))
            count = 0
            d12_avg = 0
            # y12_avg=0
            for i in range(len(train_label)):
                count += 1
                person2 = train_features[i]
                age2 = train_label[i]
                person2 = torch.Tensor(person2)
                person2 = person2.to(device)
                person2 = torch.unsqueeze(person2, 0)
                person2 = torch.unsqueeze(person2, 1)
                person2 = person2.to(torch.float32)
                d12 = net(person1, person2)
                age1_pred = age2 + d12
                d12_avg += age1_pred
            d12_avg = d12_avg.cpu()
            d12_avg = d12_avg / count
            pre_age_dist.append(d12_avg.item())
            # y12_avg=y12_avg/count
            rea_age_dist.append(age1.item())
    return train_ls, pre_age_dist, rea_age_dist


context2=np.load('/home/xule/alldata/HCP2/FC.npy')
context1=np.load('/home/xule/alldata/HCP2/age.npy',encoding='latin1')

sample_num = 600

corr_gust_list=[]
corr_gust_mark_list=[]
mse_list=[]
rmse_list=[]
mae_list=[]

# 10 rounds of 10-fold cross-validation
for lab in range(10):
    sample_list = [i for i in range(len(context1))]
    sample_list = random.sample(sample_list, sample_num)
    context2_2 = context2[sample_list, :]
    context1_1 = context1[sample_list]
    print(context2_2.shape)
    print(context1_1.shape)
    x = torch.Tensor(context2_2)
    y = torch.Tensor(context1_1)
    pre_age_dist = []
    rea_age_dist = []

    for i in range(10):
        net = SNNC()
        net = net.to(device)
        train_f, train_l, test_f, test_l = get_k_fold_data(10, i, x, y)
        fin_sample_paris_list = SampleParis(train_l, 10000)
        train_ls, pre_age_dist_fin, rea_age_dist_fin = train(net, train_f, test_f, train_l, test_l, 10000,
                                                             fin_sample_paris_list)
        print('fold %d,train rmse %f' % (i, train_ls[-1]))

    pr2 = pd.Series(pre_age_dist_fin, dtype=np.float64)
    re2 = pd.Series(rea_age_dist_fin, dtype=np.float64)
    corr_gust = round(pr2.corr(re2), 8)
    print('r:', corr_gust)
    corr_gust_mark = r2_score(re2, pr2)
    print('r2:', corr_gust_mark)
    mse = mean_squared_error(re2, pr2)
    print('MSE:', mse)
    rmse = sqrt(mse)
    print('RMSE：', rmse)
    mae = mean_absolute_error(re2, pr2)
    print('MAE：', mae)

    corr_gust_list.append(corr_gust)
    corr_gust_mark_list.append(corr_gust_mark)
    mse_list.append(mse)
    rmse_list.append(rmse)
    mae_list.append(mae)

avg_corr_gust = sum(corr_gust_list)/len(corr_gust_list)
print('avg_corr',avg_corr_gust)
avg_corr_gust_mark= sum(corr_gust_mark_list)/len(corr_gust_mark_list)
print('avg_r2_score',avg_corr_gust_mark)
avg_mse= sum(mse_list)/len(mse_list)
print('avg_mse',avg_mse)
avg_rmse= sum(rmse_list)/len(rmse_list)
print('avg_rmse',avg_rmse)
avg_mae = sum(mae_list)/len(mae_list)
print('avg_mae',avg_mae)

std_corr= np.std(corr_gust_list, ddof=1)
print('std_corr',std_corr)
std_mae= np.std(mae_list, ddof=1)
print('std_mae',std_mae)

