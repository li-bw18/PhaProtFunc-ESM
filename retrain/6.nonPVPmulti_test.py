from sklearn.metrics import precision_recall_fscore_support
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
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

batch_size=64
test_dataset = PVP(mode='test')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
nonpvpmulti = model.Model_nonPVPmulti(28)
nonpvpmulti.load_state_dict(torch.load("model/nonPVPmulti.pth"))
nonpvpmulti = nonpvpmulti.to(device)
with torch.no_grad():
    nonpvpmulti.eval()
    fo = open('metric/nonPVPmulti/prob.txt', 'w')
    fo1 = open('metric/nonPVPmulti/result.txt', 'w')
    for batch_id, batch_data in enumerate(test_dataloader):
        inputs,labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicts = torch.sigmoid(nonpvpmulti(inputs)).cpu().numpy()
        for i in range(len(predicts)):
            for j in range(27):
                fo.write(f"{predicts[i, j]}\t")
                if predicts[i, j] >= 0.5:
                    fo1.write("1\t")
                else:
                    fo1.write("0\t")
            fo.write(f"{predicts[i, 27]}\n")
            if predicts[i, 27] >= 0.5:
                fo1.write("1\n")
            else:
                fo1.write("0\n")
    fo.close()
    fo1.close()

    

