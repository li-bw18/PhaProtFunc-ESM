import os
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(description='PhaProtFunc-ESM (GPU_3)')
parser.add_argument('-o', '--output', help='Path to output directory', default=f'{os.getcwd()}/result')
parser.add_argument('-g', '--gpu_id', help='Give the IDs of GPUs you want to use [For example:"-g 0,1"], if not provided, cpu will be used', default=None)
parser.add_argument('-b', '--batch_size', type=int, help='Define the batch size used in the prediction', default=2)
args = parser.parse_args()

batch_size = args.batch_size
output = args.output

if output[-1] == '/':
    output = output[:-1]

if os.path.exists(output) is False:
    print('Error: Output directory does not exist!')
    sys.exit(1)

if args.gpu_id is not None:
    subprocess.call([f"python -u {sys.path[0]}/PVP_multi_predict.py {output} {batch_size} -g {args.gpu_id}"], shell=True)
else:
    subprocess.call([f"python -u {sys.path[0]}/PVP_multi_predict.py {output} {batch_size}"], shell=True)

if args.gpu_id is not None:
    subprocess.call([f"python -u {sys.path[0]}/nonPVP_multi_predict.py {output} {batch_size} -g {args.gpu_id}"], shell=True)
else:
    subprocess.call([f"python -u {sys.path[0]}/nonPVP_multi_predict.py {output} {batch_size}"], shell=True)

print('All processes finished for this section!')
