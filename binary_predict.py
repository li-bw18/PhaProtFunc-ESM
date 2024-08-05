import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import retrain.model as model
import utils
THRESHOLD = 0.8231529593467712
parser = argparse.ArgumentParser(description='Binary')
parser.add_argument('output', help='Path to output directory')
parser.add_argument('batch_size', type=int, help='Define the batch size used in the prediction')
parser.add_argument('-g', '--gpu_id', help='Give the IDs of GPUs you want to use [For example:"-g 0,1"], if not provided, cpu will be used', default=None)
args = parser.parse_args()

batch_size = args.batch_size
output = args.output

dataset = utils.PVP_binary(output)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

binary = model.Model_binary()
binary.load_state_dict(torch.load(f"{sys.path[0]}/model/binary.pth", map_location=device))
if device != 'cpu' and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    binary = nn.DataParallel(binary)
binary = binary.to(device)

result = pd.DataFrame(data=None)

print('Process 3: Binary prediction')
with torch.no_grad():
    binary.eval()
    all_predict = []
    all_max = []
    for batch_id, batch_data in enumerate(dataloader):
        batch_data = batch_data.to(device)
        predicts = F.softmax(binary(batch_data)[4], 1).cpu().numpy()
        all_predict.append(predicts[:, 1])
        all_max.append(np.argmax(predicts, axis=1))
    all_predict = np.concatenate(all_predict)
    all_max = np.concatenate(all_max)
    dic = {0: 'non-PVP', 1: 'PVP'}
    all_label = []
    for i in list(all_max):
        all_label.append(dic[i])
    all_confid = []
    for i in all_predict:
        if i < 0.5 or i >= THRESHOLD:
            all_confid.append('High')
        else:
            all_confid.append('Low')
    with open(f'{output}/discription.txt', 'a') as f:
        f.write('\n')
        f.write('Regardless of the prediction result of the binary task, the PVP multi-class model and the nonPVP multi-label model are both applied to all input sequences.\n')
        f.write('You could combine the predictions of each model with some prior knowledge to make the final decisions.\n')
        f.write('\n')
        f.write('File information of [binary_result.txt]:\n')
        f.write('result.txt: column 1, sequence name; column 2, predicted result; column 3, PVP probability\n')
    result['binary_pred'] = all_label
    result['PVP_prob'] = all_predict
    result['confidence'] = all_confid
    result.index = dataset.name
    result.to_csv(f'{output}/binary_result.txt', sep='\t', header=True, index=True)  
print('Process 3 finished!')
