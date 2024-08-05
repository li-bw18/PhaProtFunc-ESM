import os
import sys
import argparse
import utils
import subprocess

parser = argparse.ArgumentParser(description='PhaProtFunc-ESM')
parser.add_argument('input', help='Path to the input fasta file')
parser.add_argument('-o', '--output', help='Path to output directory', default=f'{os.getcwd()}/result')
parser.add_argument('-g', '--gpu_id', help='Give the IDs of GPUs you want to use [For example:"-g 0,1"], if not provided, cpu will be used', default=None)
parser.add_argument('-t', '--threads', type=int, help='Define the number of threads to use during blastp', default=1)
parser.add_argument('-b', '--batch_size', type=int, help='Define the batch size used in the prediction', default=2)
args = parser.parse_args()

batch_size = args.batch_size
output = args.output

if output[-1] == '/':
    output = output[:-1]

if os.path.exists(output) is False:
    subprocess.call([f"mkdir {output}"], shell=True)

print('Process 1: BLASTP annotation')

with open(f'{output}/discription.txt', 'w') as f:
    f.write('File information of [start_blastp_result.txt]:\n')
    f.write('This file saves the information of BLASTP results to the all Refseq annotated phage sequences\n')
    f.write('result.txt: column 1, sequence name; column 2, target sequence id; column 3, target sequence annotation; column 4, evalue; column 5, percent of identical amino acids\n')

subprocess.call([f'blastp -query {args.input} -out {output}/temp_blastp.out -db {sys.path[0]}/blastp_database/all/all -outfmt "6 qseqid sseqid evalue pident" -num_threads {args.threads} -evalue 1e-5'], shell=True)

utils.format_blastp(output, 0)

subprocess.call([f'rm -rf {output}/temp_blastp.out'], shell=True)

print('Process 1 finished!')

utils.token_generator(args.input, output)

if args.gpu_id is not None:
    subprocess.call([f"python -u {sys.path[0]}/binary_predict.py {output} {batch_size} -g {args.gpu_id}"], shell=True)
else:
    subprocess.call([f"python -u {sys.path[0]}/binary_predict.py {output} {batch_size}"], shell=True)

if args.gpu_id is not None:
    subprocess.call([f"python -u {sys.path[0]}/esm_embedding.py {output} {batch_size} -g {args.gpu_id}"], shell=True)
else:
    subprocess.call([f"python -u {sys.path[0]}/esm_embedding.py {output} {batch_size}"], shell=True)

if args.gpu_id is not None:
    subprocess.call([f"python -u {sys.path[0]}/PVP_multi_predict.py {output} {batch_size} -g {args.gpu_id}"], shell=True)
else:
    subprocess.call([f"python -u {sys.path[0]}/PVP_multi_predict.py {output} {batch_size}"], shell=True)

if args.gpu_id is not None:
    subprocess.call([f"python -u {sys.path[0]}/nonPVP_multi_predict.py {output} {batch_size} -g {args.gpu_id}"], shell=True)
else:
    subprocess.call([f"python -u {sys.path[0]}/nonPVP_multi_predict.py {output} {batch_size}"], shell=True)

print('Process 7: BLASTP annotation of others proteins')

with open(f'{output}/discription.txt', 'w') as f:
    f.write('File information of [others_blastp_result.txt]:\n')
    f.write('This file saves the information of BLASTP results of the "others" nonPVP proteins to the "others" Refseq annotated nonPVP sequences\n')
    f.write('result.txt: column 1, sequence name; column 2, target sequence id; column 3, target sequence annotation; column 4, evalue; column 5, percent of identical amino acids\n')

utils.get_others_list(args.input, output)

subprocess.call([f'blastp -query {output}/temp_others.fasta -out {output}/temp_blastp.out -db {sys.path[0]}/blastp_database/others/others -outfmt "6 qseqid sseqid evalue pident" -num_threads {args.threads} -evalue 1e-5'], shell=True)

utils.format_blastp(output, 1)

subprocess.call([f'rm -rf {output}/temp_blastp.out'], shell=True)

subprocess.call([f'rm -rf {output}/temp_others.fasta'], shell=True)

print('Process 7 finished!')

print('All processes finished!')
