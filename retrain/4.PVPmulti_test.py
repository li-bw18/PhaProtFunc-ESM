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
            self.x = np.array(pd.read_table("data/PVP_multi/train.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/PVP_multi/train_label.txt", header=None, index_col=None)).flatten()
        elif mode == 'valid':
            self.x = np.array(pd.read_table("data/PVP_multi/valid.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/PVP_multi/valid_label.txt", header=None, index_col=None)).flatten()
        else:
            self.x = np.array(pd.read_table("data/PVP_multi/test.txt", header=None, index_col=None))
            self.y = np.array(pd.read_table("data/PVP_multi/test_label.txt", header=None, index_col=None)).flatten()
    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y
    def __len__(self):
        return len(self.x)

batch_size=64
test_dataset = PVP(mode='test')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
pvpmulti = model.Model_PVPmulti()
pvpmulti.load_state_dict(torch.load("model/PVPmulti.pth"))
pvpmulti = pvpmulti.to(device)
with torch.no_grad():
    pvpmulti.eval()
    test_acc = 0
    test_num = 0
    fo = open('metric/PVPmulti/prob.txt', 'w')
    all_labels = np.array([])
    all_predicts = np.array([])
    for batch_id, batch_data in enumerate(test_dataloader):
        inputs,labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicts = F.softmax(pvpmulti(inputs), 1).cpu().numpy()
        for i in range(len(predicts)):
            for j in range(7):
                fo.write(f"{predicts[i, j]}\t")
            fo.write(f"{predicts[i, 7]}\n")
        predicts = np.argmax(predicts, axis=1)
        labels = labels.cpu().numpy()
        test_num += len(labels)
        test_acc += (labels == predicts).sum()
        all_labels = np.concatenate((all_labels, labels))
        all_predicts = np.concatenate((all_predicts, predicts))
    fo.close()

    pd.DataFrame(np.array([all_labels, all_predicts]).transpose(), columns=['labels', 'predict']).to_csv("metric/PVPmulti/result.txt", index=False)
    test_acc = test_acc / test_num
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(
                y_true=all_labels, y_pred=all_predicts, labels=[0, 1, 2, 3, 4, 5, 6, 7], average=None)
    
    print(f"ACC: {test_acc}")
    print(f"Macro-F1: {f_class.mean()}")
    print(f"Average precision: {p_class.mean()}")
    print(f"Average recall: {r_class.mean()}")
    print('F1_each:')
    for f in f_class:
        print(f)
    print('Recall_each:')
    for r in r_class:
        print(r)
