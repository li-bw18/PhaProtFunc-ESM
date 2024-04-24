import esm
import pandas as pd
from Bio import SeqIO
import numpy as np
import torch
from torch.utils.data import Dataset

def token_generator(file, output):
    print('Process 1: Token generation')
    alphabet = esm.pretrained.esm2_t33_650M_UR50D()[1]
    batch_converter = alphabet.get_batch_converter()
    data = []
    ret = []
    for seq_record in SeqIO.parse(file, "fasta"):
        data.append((seq_record.id, seq_record.seq))
        ret.append(seq_record.id)
    batch_tokens = pd.DataFrame(batch_converter(data)[2]).iloc[:, :1024]
    batch_tokens.index = ret
    batch_tokens.to_csv(f"{output}/tokens.txt", sep='\t', header=False, index=True)
    print('Process 1 finished!')
    return

class PVP_binary(Dataset):
    def __init__(self, output):
        super().__init__()
        self.name = pd.read_table(f"{output}/tokens.txt", header=None, index_col=0)
        self.x = np.array(self.name)
        self.name = list(self.name.index)
    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.long)
        return x
    def __len__(self):
        return len(self.x)

class PVP_multi(Dataset):
    def __init__(self, output):
        super().__init__()
        self.name = pd.read_table(f"{output}/embeddings.txt", header=None, index_col=0)
        self.x = np.array(self.name)
        self.name = list(self.name.index)
    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float)
        return x
    def __len__(self):
        return len(self.x)
