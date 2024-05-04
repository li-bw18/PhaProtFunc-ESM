import os
import argparse
import torch
from torch.utils.data import DataLoader
import utils
import esm

parser = argparse.ArgumentParser(description='3B esm embedding generation')
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

esm2 = esm.pretrained.esm2_t36_3B_UR50D()[0].to(device)

print('Process 3: 3B ESM2 embedding generation')
with torch.no_grad():
    esm2.eval()
    with open(f'{args.output}/embeddings.txt', 'w') as f:
        al = 0
        for batch_id, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            output = esm2(inputs, repr_layers=[36], return_contacts=False)["representations"][36]
            inputs = inputs.unsqueeze(-1)
            num = torch.sum(inputs>2, axis=1)
            this = output.masked_fill(inputs<=2, 0)
            sum_o = (this.sum(axis=1) / num).cpu().detach().numpy()
            for j in range(sum_o.shape[0]):
                f.write(str(dataset.name[al]))
                al += 1
                for k in range(sum_o.shape[1]):
                    f.write('\t')
                    f.write(str(sum_o[j, k]))
                f.write('\n')

print('Process 3 finished!')
