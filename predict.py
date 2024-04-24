import os
import argparse
import utils
import subprocess

parser = argparse.ArgumentParser(description='PhaProtFunc-ESM')
parser.add_argument('input', help='Path to the input fasta file')
parser.add_argument('-o', '--output', help='Path to output directory', default=f'{os.getcwd()}/result')
parser.add_argument('-g', '--gpu_id', help='Give the IDs of GPUs you want to use [For example:"-g 0,1"], if not provided, cpu will be used', default=None)
parser.add_argument('-b', '--batch_size', type=int, help='Define the batch size used in the prediction', default=2)
args = parser.parse_args()

batch_size = args.batch_size
output = args.output

if os.path.exists(output) is False:
    subprocess.call([f"mkdir {output}"], shell=True)

utils.token_generator(args.input, output)

if args.gpu_id is not None:
    subprocess.call([f"python -u binary_predict.py {args.output} {args.batch_size} -g {args.gpu_id}"], shell=True)
else:
    subprocess.call([f"python -u binary_predict.py {args.output} {args.batch_size}"], shell=True)

if args.gpu_id is not None:
    subprocess.call([f"python -u esm_embedding.py {args.output} {args.batch_size} -g {args.gpu_id}"], shell=True)
else:
    subprocess.call([f"python -u esm_embedding.py {args.output} {args.batch_size}"], shell=True)

if args.gpu_id is not None:
    subprocess.call([f"python -u PVP_multi_predict.py {args.output} {args.batch_size} -g {args.gpu_id}"], shell=True)
else:
    subprocess.call([f"python -u PVP_multi_predict.py {args.output} {args.batch_size}"], shell=True)

if args.gpu_id is not None:
    subprocess.call([f"python -u nonPVP_multi_predict.py {args.output} {args.batch_size} -g {args.gpu_id}"], shell=True)
else:
    subprocess.call([f"python -u nonPVP_multi_predict.py {args.output} {args.batch_size}"], shell=True)

print('All processes finished!')
