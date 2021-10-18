# Extracting Predictive Representations from Hundreds of Millions of Molecules

Source code of the paper "Extracting Predictive Representations from Hundreds of Millions of Molecules" by Dong Chen, Guo-Wei Wei<sup>+</sup> and Feng Pan\*

---

![model_framework](./model_protocal.png)

---

## Requirments

OS Requirements
- CentOS Linux 7 (Core)

Python Dependencies
- setuptools (>=18.0)
- python (>=3.7)
- pytorch (>=1.2)
- rdkit (2020.03)
- biopandas (0.2.7)
- numpy (1.17.4)
- scikit-learn (0.23.2)
- scipy (1.5.2)
- pandas (0.25.3)


## Installation Guide

Install from Github

```shell
# download entire folder, which named BTcode
cd BTcode/bt_pro
python setup.py build_ext --inplace
mv ./bt_pro/fairseq/data/* ./fairseq/data/
```

which should install in about 60 seconds.

## Downloading Models

Pre-trained models are publicly available.

```shell
# Pre-trained model
wget https://weilab.math.msu.edu/Downloads/chembl_pubchem_zinc_models.zip ./examples/models/
```

## Pre-training settings

The pre-training used in this work is the combination of ChEMBL27, PubChem, and ZINC13 3D datasets, which is publicly available at https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_27/, https://ftp.ncbi.nih.gov/pubchem/Compound/, and http://files.docking.org/3D/, respectively.

```shell
# Suppose the file name of the pre-training data are chembl27_train.smi and chembl27_valid.smi
# First pre-processing
python "./bt_pro/preprocess.py" --only-source --trainpref "chembl27_train.smi" --validpref "chembl27_valid.smi" --destdir "./examples/data/chembl27/" --trainoutf "train" --validoutf "valid"  --workers 20 --file-format smiles

# Pre-training command.
# Suppose the pre-processed data are stored in the ./examples/data/chembl27/
CUDA_VISIBLE_DEVICES=0 python "./bt_pro/train.py" "./examples/data/chembl27/" --train-subset "train" --valid-subset "valid" --save-dir "./examples/models/" --task masked_lm --arch roberta_base --encoder-attention-heads 8 --encoder-embed-dim 512 --encoder-ffn-embed-dim 1024 --encoder-layers 8 --dropout 0.1 --attention-dropout 0.1  --criterion masked_lm --sample-break-mode complete --tokens-per-sample 256 --skip-invalid-size-inputs-valid-test --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-6 --clip-norm 0.0 --lr-scheduler polynomial_decay --lr 0.0001 --weight-decay 0.1 --warmup-updates 5000 --total-num-update 1000000 --max-update $1000000 --save-interval 2 --save-interval-updates 100000 --log-format simple --log-interval 2000 --max-sentences 64 --update-freq 2 --ddp-backend no_c10d --fp16 --reset-optimizer --reset-dataloader --reset-meters

# the pre-trained model will save as ./examples/data/chembl27/checkpoint_best.pt
```

## Reproduction instructions

- All parameter settings for the training process can be referred to the "Methods" section in the artical.

## Customize task-specific fingerprints

For users who want to build a new task-specific model from a set of molecules with corresponding properties, here we provide script for generating the fingerprints. By default, we use supervised learning-based strategy to fine-tune the pre-trained model and the regression task was used. The following steps need to be performed on a platform that supports GPU computing.

```shell
# Generate Bidirectional Transformer-based Fingerprints

# step 1, download the pre-trained model
wget https://weilab.math.msu.edu/Downloads/chembl_pubchem_zinc_models.zip ./examples/models/
# unzip the .zip file to target folder

# step 2, pre-process input data (Binarize the input data to speed up the training.)
python "./bt_pro/preprocess.py" --only-source --trainpref "./examples/data/example_train_canonical.smi" --validpref "./examples/data/example_valid_canonical.smi" --destdir "./examples/data/input0/" --trainoutf "train" --validoutf "valid"  --workers 20 --file-format smiles --srcdict "./examples/models/dict.txt"

# step 3, fine-tuning the pre-trained model
train_data_len=358
num_epoch=50
num_sent_pergpu=16
updata_freq=1
num_warmup=`expr $num_epoch \* $train_data_len / ${num_sent_pergpu} / $updata_freq / 10 `
max_num_update=100000
CUDA_VISIBLE_DEVICES=0 python "./bt_pro/train.py" "./examples/data/" --save-dir "./examples/models/" --train-subset train --valid-subset valid --restore-file "./examples/models/checkpoint_pretrained.pt" --task sentence_prediction --num-classes 1 --regression-target --init-token 0 --best-checkpoint-metric loss --arch roberta_base --bpe smi --encoder-attention-heads 8 --encoder-embed-dim 512 --encoder-ffn-embed-dim 1024 --encoder-layers 8 --dropout 0.1 --attention-dropout 0.1 --criterion sentence_prediction --max-positions 256 --truncate-sequence --skip-invalid-size-inputs-valid-test --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-6 --clip-norm 0.0 --lr-scheduler polynomial_decay --lr 0.0001 --warmup-updates ${num_warmup} --total-num-update  ${max_num_update} --max-update ${max_num_update} --max-epoch ${num_epoch} --weight-decay 0.1 --log-format simple --reset-optimizer --reset-dataloader --reset-meters --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state --find-unused-parameters --log-interval 5 --max-sentences ${num_sent_pergpu} --update-freq ${updata_freq} --required-batch-size-multiple 1 --ddp-backend no_c10d 

# step 4, generate BT-FPs
python "./bt_pro/generate_bt_fps.py" --model_name_or_path "./examples/models/" --checkpoint_file "checkpoint.pt" --data_name_or_path  "./examples/data/input0/" --dict_file "./example/data/input0/dict.txt" --target_file "./examples/data/example_train_canonical.smi" --save_feature_path "./examples/examples_bt_train_features.npy"
python "./bt_pro/generate_bt_fps.py" --model_name_or_path "./examples/models/" --checkpoint_file "checkpoint.pt" --data_name_or_path  "./examples/data/input0/" --dict_file "./example/data/input0/dict.txt" --target_file "./examples/data/example_valid_canonical.smi" --save_feature_path "./examples/examples_bt_valid_features.npy"

```

For the data in the example, the entire process took less than 30 minutes.

## License

All codes released in this study is under the MIT License.
