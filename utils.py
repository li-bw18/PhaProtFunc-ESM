import esm
import pandas as pd
from Bio import SeqIO
import numpy as np
import torch
from torch.utils.data import Dataset
import sys

def token_generator(file, output):
    print('Process 2: Token generation')
    alphabet = esm.pretrained.esm2_t33_650M_UR50D()[1]
    batch_converter = alphabet.get_batch_converter()
    data = []
    ret = []
    for seq_record in SeqIO.parse(file, "fasta"):
        data.append((seq_record.id, str(seq_record.seq).replace('J', 'X')))
        ret.append(seq_record.id)
    batch_tokens = pd.DataFrame(batch_converter(data)[2]).iloc[:, :1024]
    batch_tokens.index = ret
    batch_tokens.to_csv(f"{output}/tokens.txt", sep='\t', header=False, index=True)
    print('Process 2 finished!')
    return

def format_blastp(output, typ):
    info = {}
    with open(f'{sys.path[0]}/blastp_database/info_all.txt') as f:
        while 1:
            line = f.readline()
            if line == '':
                break
            line = line[:-1].split('\t')
            info[line[0]] = line[1]
    out = {}
    if typ == 0:
        path = f'{output}/start_blastp_result.txt'
        fo = open(path, 'w')
    else:
        path = f'{output}/others_blastp_result.txt'
        fo = open(path, 'w')
    fo.write('query\ttarget\tannotation\tevalue\tidentity\n')
    with open(f'{output}/temp_blastp.out') as f:
        while 1:
            line = f.readline()
            if line == '':
                break
            line = line[:-1].split('\t')
            if line[0] not in out.keys():
                if float(line[2]) <= 1e-5 and float(line[3]) >= 30:
                    out[line[0]] = 1
                    fo.write(f"{line[0]}\t{line[1]}\t{info[line[1]]}\t{line[2]}\t{line[3]}\n")

    fo.close()

def get_others_list(file, output):
    fi1 = open(f'{output}/binary_result.txt')
    fi1.readline()
    fi2 = open(f'{output}/nonPVP_multi_result.txt')
    fi2.readline()
    fo = open(f'{output}/temp_others.fasta', 'w')
    for seq_record in SeqIO.parse(file, "fasta"):
        line1 = fi1.readline().split('\t')[1]
        line2 = fi2.readline().split('\t')[1]
        if line1 == 'non-PVP' and 'others' in line2:
            fo.write(f">{seq_record.id}")
            fo.write('\n')
            fo.write(str(seq_record.seq))
            fo.write('\n')

    fi1.close()
    fi2.close()
    fo.close()

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
