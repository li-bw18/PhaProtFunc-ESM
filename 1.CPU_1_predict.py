import os
import sys
import argparse
import utils
import subprocess

parser = argparse.ArgumentParser(description='PhaProtFunc-ESM (CPU_1)')
parser.add_argument('input', help='Path to the input fasta file')
parser.add_argument('-o', '--output', help='Path to output directory', default=f'{os.getcwd()}/result')
parser.add_argument('-t', '--threads', type=int, help='Define the number of threads to use during blastp', default=1)
args = parser.parse_args()

output = args.output

if output[-1] == '/':
    output = output[:-1]

if os.path.exists(output) is False:
    subprocess.call([f"mkdir {output}"], shell=True)

print('Process 1: BLASTP annotation')

with open(f'{output}/discription.txt', 'w') as f:
    f.write('File information of [start_blastp_result.txt]:\n')
    f.write('This file saves the information of BLASTP results to the all Refseq annotated phage sequences\n')
    f.write('column 1, sequence name; column 2, target sequence id; column 3, target sequence annotation; column 4, evalue; column 5, percent of identical amino acids\n')

subprocess.call([f'blastp -query {args.input} -out {output}/temp_blastp.out -db {sys.path[0]}/blastp_database/all/all -outfmt "6 qseqid sseqid evalue pident" -num_threads {args.threads} -evalue 1e-5'], shell=True)

utils.format_blastp(output, 0)

subprocess.call([f'rm -rf {output}/temp_blastp.out'], shell=True)

print('Process 1 finished!')

utils.token_generator(args.input, output)

print('All processes finished for this section!')
