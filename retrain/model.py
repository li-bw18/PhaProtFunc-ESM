import torch.nn as nn
import torch.nn.functional as F
import torch
import esm

class Model_binary(nn.Module):
    def __init__(self, lis=['30', '31', '32']):
        super().__init__()
        self.esm = esm.pretrained.esm2_t33_650M_UR50D()[0]
        for na, p in self.named_parameters():
            sp = na.split('.')
            if len(sp) > 2 and sp[2] not in lis:
                p.requires_grad = False
            else:
                p.requires_grad = True
        self.fc1 = nn.Linear(1280, 320)
        self.fc2 = nn.Linear(320, 80)
        self.fc3 = nn.Linear(80, 20)
        self.fc4 = nn.Linear(20, 2)
    def forward(self, batch_tokens):
        x = self.esm(batch_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
        batch_tokens = batch_tokens.unsqueeze(-1)
        x = x.masked_fill(batch_tokens<=2, 0)
        num = torch.sum(batch_tokens>2, axis=1)
        x = torch.sum(x, dim=1) / num
        ret = []
        ret.append(x)
        x = F.relu(self.fc1(x))
        ret.append(x)
        x = F.relu(self.fc2(x))
        ret.append(x)
        x = F.relu(self.fc3(x))
        ret.append(x)
        ret.append(self.fc4(x))
        return ret

class Model_nonPVPmulti(nn.Module):
    def __init__(self, n_label):
        super().__init__()
        self.fc1 = nn.Linear(2560, 1280)
        self.fc2 = nn.Linear(1280, 640)
        self.fc3 = nn.Linear(640, 160)
        self.fc4 = nn.Linear(160, 40)
        self.fc5 = nn.Linear(40, n_label)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)

class Model_PVPmulti(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2560, 1280)
        self.bn1 = nn.BatchNorm1d(1280)
        self.fc2 = nn.Linear(1280, 320)
        self.bn2 = nn.BatchNorm1d(320)
        self.fc3 = nn.Linear(320, 80)
        self.bn3 = nn.BatchNorm1d(80)
        self.fc4 = nn.Linear(80, 20)
        self.bn4 = nn.BatchNorm1d(20)
        self.fc5 = nn.Linear(20, 8)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        return self.fc5(x)
