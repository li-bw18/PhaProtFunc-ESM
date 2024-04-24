import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, precision_recall_curve, average_precision_score
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

batch_size = 16
test_dataset = PVP(mode='test')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
binary = model.Model_binary()
binary.load_state_dict(torch.load("model/binary.pth"))
binary = binary.to(device)
all_label = np.array([])
all_predict = np.array([])
all_max = np.array([])

with torch.no_grad():
    binary = binary.eval()
    for batch_id, batch_data in enumerate(test_dataloader):
        inputs,labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicts = F.softmax(binary(inputs)[4], 1).cpu().numpy()
        all_label = np.concatenate((all_label, labels.cpu().numpy()))
        all_predict = np.concatenate((all_predict, predicts[:, 1]))
        all_max = np.concatenate((all_max, np.argmax(predicts, axis=1)))
    
    pd.DataFrame(np.array([all_label, all_max]).transpose(), columns=['labels', 'predict']).to_csv("metric/binary/result.txt", index=False)
    test_acc = (all_label == all_max).sum() / len(all_label)
    test_auc = roc_auc_score(all_label, all_predict)
    print(f"ACC:{test_acc}")
    print(f"AUC:{test_auc}")
    fpr, tpr, thersholds = roc_curve(all_label, all_predict)
    p_class, r_class, f_class, _ = precision_recall_fscore_support(y_true=all_label, y_pred=all_max, average='binary')
    print(f"Precision:{p_class}")
    print(f"Recall:{r_class}")
    print(f"F1:{f_class}")
    pd.DataFrame(fpr).to_csv("metric/binary/fpr.txt", sep='\t', index=False, header=False)
    pd.DataFrame(tpr).to_csv("metric/binary/tpr.txt", sep='\t', index=False, header=False)
    test_ap = average_precision_score(all_label, all_predict)
    print(f"AP:{test_ap}")
    precision, recall, thersholds = precision_recall_curve(all_label, all_predict)
    pd.DataFrame(precision).to_csv("metric/binary/precision.txt", sep='\t', index=False, header=False)
    pd.DataFrame(recall).to_csv("metric/binary/recall.txt", sep='\t', index=False, header=False)

