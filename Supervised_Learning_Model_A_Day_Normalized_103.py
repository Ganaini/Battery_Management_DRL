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
import torch.nn.functional as F



from Env_DQN_MG_107 import T

HIDDEN_N = 256
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 50000

data_path = r"E:\Mahmoud\PhD_Work\Presentations\March_25_Forecast\A_day_Training.csv"
df = pd.read_csv(data_path)

y_train = df['Action']
X_train = df[['Price', 'PV', 'Load', 'SOC']]

# Normalize state space to match DRL input
X_train['Price'] = X_train['Price'] / max(X_train['Price'])
X_train['PV'] = X_train['PV'] / max(X_train['PV'])
X_train['Load'] = X_train['Load'] / max(X_train['Load'])


class Net(nn.Module):
    def __init__(self, input_n, hidden_n, output_n):
        super(Net, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_n, hidden_n),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_n, hidden_n),
            nn.ReLU(),
            nn.Dropout(p=0.1),  
            nn.Linear(hidden_n, hidden_n),
            nn.ReLU(),
            nn.Dropout(p=0.1),                   
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

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        states, labels = batch

        preds = model(states)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds


if __name__ == '__main__':
    
    model = Net(4, HIDDEN_N, 3)
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr = LR)
    
    X_train_norm = preprocessing.normalize(X_train)
    
    input_v = torch.tensor(X_train_norm, dtype=torch.float32)  
    label_v = torch.tensor(y_train, dtype=torch.long)    
    
    train_set = MyDataset(input_v, label_v)
    # check trainset length
    # print(len(train_set))
    
    # How many of each label exists in the dataset:
    # print(train_set.targets.bincount())
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)


    for epoch in range(EPOCHS):
        total_loss = 0
        total_correct = 0
        c = 0
        # batch = next(iter(train_loader)) # Getting a batch
        for batch in train_loader: # Get Batch
            states, labels = batch
            
            preds = model(states)
            loss = F.cross_entropy(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)    
        
        accuracy = total_correct / len(train_set) 
        if accuracy > 0.99:
            break
 

        print("epoch: %d  total_correct: %d  training loss: %.3f  Training Accuracy: %.3f" % (
            epoch, total_correct, total_loss, accuracy))

    with torch.no_grad():
        prediction_loader = DataLoader(train_set, batch_size=T)
        train_preds = get_all_preds(model, prediction_loader)
        preds_correct = get_num_correct(train_preds, train_set.targets)
        print('\ntotal correct:', preds_correct, 'out of', len(train_preds))
        print('accuracy: %.3f' % (preds_correct / len(train_set)))
        # for itr in range(ITERATIONS):
        #     input_v = X_train
        
    # saving predictid actions into a csv file
    predicted_actions = train_preds.argmax(dim=1)   
        
    df_predictons = pd.DataFrame({'Supervised Predictions': predicted_actions})
    df_predictons.to_csv(r'E:\Mahmoud\PhD_Work\Presentations\March_25_Forecast\Supervised_Predictions_Normalized_103.csv')
    
    # saving model parameters to use in initializing the DRL model parameters    
    save_path = r'E:\Mahmoud\PhD_Work\Presentations\March_25_Forecast\Models\A_day_trained_Normalized_103.pth'
    torch.save(model.state_dict(), save_path)
        





        