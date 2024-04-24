import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import model


# Determine which GPU will be used
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 2024
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class PVP(Dataset):
    def __init__(self, mode='train'):
        super(PVP, self).__init__()
        if mode == 'train':
            self.x = np.array(pd.read_table("data/nonPVP_multi/train.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/nonPVP_multi/train_label.txt", header=None, index_col=None))
        elif mode == 'valid':
            self.x = np.array(pd.read_table("data/nonPVP_multi/valid.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/nonPVP_multi/valid_label.txt", header=None, index_col=None))
        else:
            self.x = np.array(pd.read_table("data/nonPVP_multi/test.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/nonPVP_multi/test_label.txt", header=None, index_col=None))
    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float)
        y = torch.tensor(self.y[index], dtype=torch.float)
        return x, y
    def __len__(self):
        return len(self.x)

def trail(model, train_dataloader, val_dataloader, criterion, optimizer, nepoch, path):
    current_lowest_loss = 1e9
    current_best_epoch = 0
    train_loss_list = []
    valid_loss_list = []
    valid_acc_list = []
    for epoch_id in range(nepoch):
        model.train()
        train_loss = 0
        for batch_id, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs,labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            predicts = torch.sigmoid(model(inputs))
            loss = criterion(predicts, labels)
            train_loss += loss.cpu().detach().numpy()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_dataloader)
        train_loss_list.append(train_loss)
        valid_loss = 0 
        valid_acc = 0
        valid_num = 0
        with torch.no_grad():
            model.eval()
            for batch_id, batch_data in enumerate(val_dataloader):
                inputs,labels = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)
                predicts = torch.sigmoid(model(inputs))
                valid_loss += criterion(predicts, labels).cpu().detach().numpy()
                predicts = predicts.cpu().numpy()
                valid_num += len(labels) * 13
                valid_acc += (labels.cpu().numpy() == (predicts>=0.5)).sum()
        valid_loss = valid_loss / len(val_dataloader)
        valid_loss_list.append(valid_loss)
        valid_acc = valid_acc / valid_num
        if valid_loss < current_lowest_loss:
            current_lowest_loss = valid_loss
            current_best_epoch = epoch_id
            torch.save(model.state_dict(), path)
        print("Epoch:", epoch_id)
        print("--------------------") 
        print("Train loss:", train_loss)
        print("Valid loss:", valid_loss)
        print("Valid ACC:", valid_acc)
        print("--------------------") 
        if epoch_id - current_best_epoch == 10:
            break
    print("Best Valid loss:", current_lowest_loss)  
    print("Best Epoch:", current_best_epoch)          
    print("--------------------") 
    return train_loss_list, valid_loss_list, valid_acc_list 

batch_size = 64
learning_rate = 5e-5
weight_decay = 1e-5
nepoch = 200

nonpvpmulti = model.Model_nonPVPmulti(13)
nonpvpmulti = nonpvpmulti.to(device)
train_dataset = PVP(mode='train')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = PVP(mode='valid')
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(nonpvpmulti.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_loss_list, valid_loss_list, valid_acc_list = trail(nonpvpmulti, train_dataloader, val_dataloader, criterion, optimizer, nepoch, "model/nonPVPmulti.pth")

