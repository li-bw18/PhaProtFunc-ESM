import os
import sys
import argparse
import utils
import subprocess

parser = argparse.ArgumentParser(description='PhaProtFunc-ESM (CPU_2)')
parser.add_argument('input', help='Path to the input fasta file')
parser.add_argument('-o', '--output', help='Path to output directory', default=f'{os.getcwd()}/result')
parser.add_argument('-t', '--threads', type=int, help='Define the number of threads to use during blastp', default=1)
args = parser.parse_args()

output = args.output

if output[-1] == '/':
    output = output[:-1]

if os.path.exists(output) is False:
    print('Error: Output directory does not exist!')
    sys.exit(1)

print('Process 7: BLASTP annotation of others proteins')

with open(f'{output}/discription.txt', 'a') as f:
    f.write('\n')
    f.write('File information of [others_blastp_result.txt]:\n')
    f.write('This file saves the information of BLASTP results of the "others" nonPVP proteins to the "others" Refseq annotated nonPVP sequences\n')
    f.write('column 1, sequence name; column 2, target sequence id; column 3, target sequence annotation; column 4, evalue; column 5, percent of identical amino acids\n')

utils.get_others_list(args.input, output)

subprocess.call([f'blastp -query {output}/temp_others.fasta -out {output}/temp_blastp.out -db {sys.path[0]}/blastp_database/others/others -outfmt "6 qseqid sseqid evalue pident" -num_threads {args.threads} -evalue 1e-5'], shell=True)

utils.format_blastp(output, 1)

subprocess.call([f'rm -rf {output}/temp_blastp.out'], shell=True)

subprocess.call([f'rm -rf {output}/temp_others.fasta'], shell=True)

print('Process 7 finished!')

utils.left_join_blastp(output)

utils.summary(output)

subprocess.call([f'rm -rf {output}/add_na_start_blastp_result.txt'], shell=True)

subprocess.call([f'rm -rf {output}/add_na_others_blastp_result.txt'], shell=True)

with open(f'{output}/discription.txt', 'a') as f:
    f.write('\n')
    f.write('File information of [summary_result.txt]:\n')
    f.write('This file saves the major results of the whole pipeline\n')
    f.write('column 1, sequence name; column 2, blastp annotation; column 3, binary prediction result; column 4, binary prediction confidence; column 5, PVP detailed annotation (when column 3=PVP); column 6, PVP annotation confidence; column 7, nonPVP detailed annotation (when column 3=nonPVP); column 8, nonPVP annotation confidence; column 9, others blastp annotation (when others in column 7)\n')

print('All processes finished!')
