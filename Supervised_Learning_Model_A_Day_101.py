# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:12:25 2021

@author: mahmo
"""

from sklearn import preprocessing
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
# from torchvision import transforms
import numpy as np

HIDDEN_N = 128
LR = 0.001
BATCH_SIZE = 32
ITERATIONS = 50000

data_path = r"D:\presentations\March_25_Forecast\A_day_Training.csv"
df = pd.read_csv(data_path)

y_train = df['Action']
X_train = df[['Price', 'PV', 'Load', 'SOC']]


class Net(nn.Module):
    def __init__(self, input_n, hidden_n, output_n):
        super(Net, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_n, hidden_n),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_n, hidden_n),
            nn.ReLU(),
            # nn.Dropout(),            
            nn.Linear(hidden_n, output_n)
            )
        
    def forward(self, x):
        return self.model(x)
    

class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        self.targets = y


    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.targets[index]



if __name__ == '__main__':
    
    model = Net(4, HIDDEN_N, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr = LR)
    
    X_train_norm = preprocessing.normalize(X_train)
    
    input_v = torch.tensor(X_train_norm, dtype=torch.double)  
    label_v = torch.tensor(y_train, dtype=torch.uint8)    
    
    train_set = MyDataset(input_v, label_v)
    # check trainset length
    # print(len(train_set))
    
    # How many of each label exists in the dataset:
    # print(train_set.targets.bincount())
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)


    # for itr in range(ITERATIONS):
    #     input_v = X_train





        