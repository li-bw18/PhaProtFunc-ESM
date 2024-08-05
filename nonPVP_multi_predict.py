import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import retrain.model as model
import utils
THRESHOLD = [0.8994660377502441, 0.9613317847251892, 0.8965098857879639, 0.6907848119735718, 0.9115489721298218, 0.8960577845573425, 
             0.9361218214035034, 0.9714311361312866, 0.8769540786743164, 0.8358472585678101, 0.7862973213195801, 0.879717230796814, 
             0.9982032775878906, 0.7012633681297302, 0.7662699222564697, 0.8361528515815735, 0.7183528542518616, 0.9357391595840454, 
             0.9885557293891908, 0.8885876536369324, 0.905609667301178, 0.9553091526031494, 0.7482558488845825, 0.991552233695984, 
             0.8247019052505493, 0.5, 0.9828835129737854, 0.9880273342132568]
parser = argparse.ArgumentParser(description='nonPVPmulti')
parser.add_argument('output', help='Path to output directory')
parser.add_argument('batch_size', type=int, help='Define the batch size used in the prediction')
parser.add_argument('-g', '--gpu_id', help='Give the IDs of GPUs you want to use [For example:"-g 0,1"], if not provided, cpu will be used', default=None)
args = parser.parse_args()

batch_size = args.batch_size
output = args.output

dataset = utils.PVP_multi(output)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

nonpvpmulti = model.Model_nonPVPmulti(28)
nonpvpmulti.load_state_dict(torch.load(f"{sys.path[0]}/model/nonPVPmulti.pth"))
if device != 'cpu' and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    nonpvpmulti = nn.DataParallel(nonpvpmulti)
nonpvpmulti = nonpvpmulti.to(device)

result = pd.DataFrame(data=None)

print('Process 6: nonPVP multi-label prediction')
with torch.no_grad():
    dic = {0: 'nuclease', 1: 'endonuclease', 2: 'polymerase', 3: 'terminase', 4: 'transferase',
           5: 'lysin', 6: 'exonuclease', 7: 'helicase', 8: 'holin/anti-holin', 9: 'reductase', 
           10: 'primase', 11: 'kinase', 12: 'methyltransferase', 13: 'ligase', 14: 'hydrolase', 
           15: 'synthase', 16: 'integrase', 17: 'esterase', 18: 'atpase', 19: 'isomerase', 
           20: 'phosphatase', 21: 'phosphoesterase', 22: 'peptidase', 23: 'phosphohydrolase', 
           24: 'protease', 25: 'topoisomerase', 26: 'anti-repressor', 27: 'others'}
    nonpvpmulti.eval()
    class_prob = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[], 20:[], 21:[], 22:[], 23:[], 24:[], 25:[], 26:[], 27:[]}
    all_max = []
    all_confid = []
    for batch_id, batch_data in enumerate(dataloader):
        batch_data = batch_data.to(device)
        predicts = torch.sigmoid(nonpvpmulti(batch_data)).cpu().numpy()
        for i in range(28):
            class_prob[i].append(predicts[:, i])
        for i in range(predicts.shape[0]):
            ch_re = ''
            confid = ''
            for j in range(predicts.shape[1]):
                if predicts[i, j] > 0.5:
                    ch_re += f'{dic[j]},'
                    if predicts[i, j] >= THRESHOLD[j]:
                        confid += 'High,'
                    else:
                        confid += 'Low,'
            if ch_re != '':
                all_max.append(ch_re[:-1])  
                all_confid.append(confid[:-1]) 
            else:
                all_max.append('NA')
                all_confid.append('NA')
    with open(f'{output}/discription.txt', 'a') as f:
        f.write('\n')
        f.write('File information of [non_PVP_multi_result.txt]:\n')
        f.write('result.txt: column 1, sequence name; column 2, predicted result; column 3-30, probability\n')
    result['nonPVP_multi_pred'] = all_max
    for i in range(28):
        result[f'{dic[i]} probability'] = np.concatenate(class_prob[i])
    result['confidence'] = all_confid
    result.index = dataset.name
    result.to_csv(f'{output}/nonPVP_multi_result.txt', sep='\t', header=True, index=True)  
print('Process 6 finished!')
