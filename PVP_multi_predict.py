import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import model
import utils
THRESHOLD = [0.2427910417318344, 0.9295542240142822, 0.6040518879890442, 0.6302810907363892, 
             0.7795376181602478, 0.6585505604743958, 0.8244971632957458, 0.9907620549201964]
parser = argparse.ArgumentParser(description='PVPmulti')
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

pvpmulti = model.Model_PVPmulti()
pvpmulti.load_state_dict(torch.load(f"{sys.path[0]}/model/PVPmulti.pth", map_location=device))
if device != 'cpu' and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    pvpmulti = nn.DataParallel(pvpmulti)
pvpmulti = pvpmulti.to(device)

result = pd.DataFrame(data=None)

print('Process 5: PVP multi-class prediction')
with torch.no_grad():
    pvpmulti.eval()
    class_prob = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
    all_max = []
    for batch_id, batch_data in enumerate(dataloader):
        batch_data = batch_data.to(device)
        predicts = F.softmax(pvpmulti(batch_data), 1).cpu().numpy()
        for i in range(8):
            class_prob[i].append(predicts[:, i])
        all_max.append(np.argmax(predicts, axis=1))
    all_max = np.concatenate(all_max)
    dic = {0: 'major head', 1: 'minor head', 2: 'neck', 3: 'major tail', 
           4: 'minor tail', 5: 'tail sheath', 6: 'baseplate', 7: 'tail fiber'}
    all_label = []
    for i in list(all_max):
        all_label.append(dic[i])
    with open(f'{output}/discription.txt', 'a') as f:
        f.write('\n')
        f.write('File information of [PVP_multi_result.txt]:\n')
        f.write('column 1, sequence name; column 2, predicted result; column 3-10, probability\n')
    result['PVP_multi_pred'] = all_label
    for i in range(8):
        result[f'{dic[i]} probability'] = np.concatenate(class_prob[i])
    all_confid = []
    for i in range(len(result)):
        if result.iloc[i, all_max[i]+1] >= THRESHOLD[all_max[i]]:
            all_confid.append('High')
        else:
            all_confid.append('Low')
    result['confidence'] = all_confid
    result.index = dataset.name
    result.to_csv(f'{output}/PVP_multi_result.txt', sep='\t', header=True, index=True)  
print('Process 5 finished!')
