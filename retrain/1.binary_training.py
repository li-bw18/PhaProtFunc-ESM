import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
from sklearn.metrics import roc_auc_score
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
            self.x = np.array(pd.read_table("data/binary/train.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/binary/train_label.txt", header=None, index_col=None)).flatten()
        elif mode == 'valid':
            self.x = np.array(pd.read_table("data/binary/valid.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/binary/valid_label.txt", header=None, index_col=None)).flatten()
        else:
            self.x = np.array(pd.read_table("data/binary/test.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/binary/test_label.txt", header=None, index_col=None)).flatten()
    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.long)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y
    def __len__(self):
        return len(self.x)

def trail(model, train_dataloader, val_dataloader, criterion, optimizer, nepoch, path):
    current_lowest_loss = 1e9
    current_best_epoch = 0
    train_loss_list = []
    valid_loss_list = []
    valid_auc_list = []
    for epoch_id in range(nepoch):
        model = model.train()
        train_loss = 0
        for batch_id, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs,labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss = criterion(model(inputs)[4], labels)
            train_loss += loss.cpu().detach().numpy()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_dataloader)
        train_loss_list.append(train_loss)
        valid_loss = 0 
        all_label = np.array([])
        all_predict = np.array([])
        with torch.no_grad():
            model = model.eval()
            for batch_id, batch_data in enumerate(val_dataloader):
                inputs,labels = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)
                predicts = model(inputs)[4]
                valid_loss += criterion(predicts, labels).cpu().detach().numpy()
                predicts = F.softmax(predicts, 1).cpu().numpy()
                all_label = np.concatenate((all_label, labels.cpu().numpy()))
                all_predict = np.concatenate((all_predict, predicts[:, 1]))
        valid_loss = valid_loss / len(val_dataloader)
        valid_loss_list.append(valid_loss)
        valid_auc = round(roc_auc_score(all_label, all_predict), 4)
        valid_auc_list.append(valid_auc)
        if valid_loss < current_lowest_loss:
            current_lowest_loss = valid_loss
            current_best_epoch = epoch_id
            torch.save(model.state_dict(), path)
        print("Epoch:", epoch_id)
        print("--------------------") 
        print("Train loss:", train_loss)
        print("Valid loss:", valid_loss)
        print("Valid AUC:", valid_auc)
        print("--------------------") 
        if epoch_id - current_best_epoch == 2:
            break
    print("Best Valid loss:", current_lowest_loss)  
    print("Best Epoch:", current_best_epoch)          
    print("--------------------") 
    return train_loss_list, valid_loss_list, valid_auc_list


batch_size = 16
learning_rate = 2e-5
weight_decay = 1e-5
nepoch = 10

binary = model.Model_binary()
binary = binary.to(device)
train_dataset = PVP(mode='train')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = PVP(mode='valid')
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(binary.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_loss_list, valid_loss_list, valid_auc_list = trail(binary, train_dataloader, val_dataloader, criterion, optimizer, nepoch, "model/binary.pth")
