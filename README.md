# PhaProtFunc-ESM (Phage protein annotation with a pretrained protein language model and an MLP neural network) #
 
## Description ##

A pipeline for the annotation of phage proteins.

Two steps of annotation (Both the tradtional alignment and the AI-based prediction):

1 BLASTP to RefSeq phage proteins with known functions (results in 'start_blastp_result.txt')

2 Annotation based on deep learning models

(1) Binary: PVP (phage virion proteins, structural proteins) or non-PVP

(2-1) Multi-class prediction of PVP: major head, minor head, neck, major tail, minor tail, tail sheath, baseplate, tail fiber

(2-2) Multi-label prediction of non-PVP: endonuclease, polymerase, terminase, helicase, lysin, exonuclease, reductase, holin, kinase, methyltransferase, primase, ligase, synthase, integrase, hydrolase, others (use BLASTP to further annotate 'others' proteins, and the results are in 'others_blastp_result.txt')

## Steps to predict ##

### Step 1: prepare the environment ###

(1) Download the PhaProtFunc-ESM software from github

`git clone https://github.com/li-bw18/PhaProtFunc-ESM.git`

(2) Go to the directory of PhaProtFunc-ESM

`cd PhaProtFunc-ESM` (for example)

(3) Create a new conda environment 

`conda create -n esmphage`

(4) Enter the conda environment 

`conda activate esmphage`

(5) Install the following software

a. pytorch: 

If you want to use the CPU version, please run `conda install pytorch torchvision torchaudio cpuonly -c pytorch`.

If you want to use the GPU version, please go to https://pytorch.org/get-started and get the conda or pip install command according to your device and demand.

b. fair-esm: `pip install fair-esm`

c. pandas: `conda install pandas`

d. biopython: `conda install -c bioconda biopython`

e. blastp: `conda install -c bioconda blast`

### Step 2: download the trained model ###

(1) Download the model (The working path is still 'PhaProtFunc-ESM')

`wget -c https://zenodo.org/records/12797951/files/model.tar.gz`

(2) Unpack the file

`tar xzvf model.tar.gz`

### (Optional) Step 3: test the software ###

Run this command (a test prediction with 20 proteins) to see whether the software has installed correctly.

`python predict.py test.fa -o result/`

If the software is installed correctly and completely, this step will finish in less than 10 minutes (might be longer if your device is too old) without any error. The results of the test prediction will be saved in the result folder.

### Step 4: prediction ###

(1) Preparations

Your proteins in a fasta file (path: ???.fa).

A directory to save the output files (path: ???/).

If you want to use GPU(s), please prepare the IDs of the GPU(s) you want to use, for example, a single-GPU machine, here it is prepared to be 0; multi-GPU machine using only one GPU, here it is prepared as x (x is the GPU ID used); multi-GPU machine using multiple GPUs, here it is prepared as x1,x2,... (x1,x2,... are the GPU IDs you want to use).

(2) Prediction

CPU: `python predict.py ???.fa -o ???/`

single GPU machine: `python predict.py ???.fa -g 0 -o ???/` 

multi GPU machine, using one GPU: `python predict.py ???.fa -g x -o ???/` 

multi GPU machine, using multi GPUs: `python predict.py ???.fa -g x1,x2,... -o ???/` 

-o determines the output directory, -g determines the IDs of GPUs you want to use (not given -g, will use CPU)

If you want to change the batch size (default is 2), please use -b. Note that the batch size cannot be negative and should not be smaller than the number of GPUs used.

If you want to change the number of threads used during the BLASTP (default is 1), please use -t (eg. '-t 16' for 16 threads).

Example commands:

Predict proteins in 'example.fasta', save the results to 'result/', batch size is 16, and 32 threads for BLASTP.

CPU: `python predict.py example.fa -o result/ -b 16 -t 32`

single GPU machine: `python predict.py example.fa -o result/ -g 0 -b 16 -t 32` 

multi GPU machine, using one GPU (ID:2): `python predict.py example.fa -o result/ -g 2 -b 16 -t 32` 

multi GPU machine, using eight GPUs (ID:0-7): `python predict.py example.fa -o result/ -g 0,1,2,3,4,5,6,7 -b 16 -t 32` 

The descriptions for the result files are in the 'discription.txt' file of the output directory.
