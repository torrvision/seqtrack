# Data

We use the `trackdat` repository to manage data.
Refer to `seqtrack.data` and `seqtrack.train.setup_data` for more detail.

Note that we store a cached index of each dataset in `data_cache_dir`.
This should be a single location that is shared by all instances of the program.
This cache is generated if it does not exist.
Caution: This may lead to a race condition if jobs are submitted in parallel (and the cache does not exist).

We support multiple pre-processed versions of each dataset.
The default is `original`.
This is the version which is used to produce the cachced index.
To use another pre-processed instance of a dataset:
```bash
    --preproc=resize_force_360x360 \
```

It is possible to untar the data at the start of the job.
To untar to a fixed location:
```bash
    --untar \
    --tmp_data_dir="/tmp/seqtrack/data" \
```
This is particularly useful for SLURM, to create a local copy of the dataset.
It is generally fast to copy a single large tar file over the network, but slow to access each individual image.
If you are using SLURM, `tmp_data_dir` will be overridden in `seqtrack/tools/train_work.py`.

During development, we often work on a single machine and we do not want to untar the dataset every time we train.
Then remove `--untar` and simply do:
```bash
    --data_dir="/tmp/seqtrack/data" \
```
(Note that you can run the program once with `--untar` and `--tmp_data_dir` to set up the dataset.)


# Training

To train a model:
```bash
python -m seqtrack.tools.train \
    --num_trials=1 \
    --imwidth=360 \
    --imheight=360 \
    --eval_datasets otb_50 \
    --eval_samplers full \
    --max_eval_videos=100 \
    --period_assess=50000 \
    --period_ckpt=10000 \
    --num_steps=100000 \
    --optimizer=adam \
    --lr_init=1e-3 \
    --loglevel=info \
    --verbose_train
```
Caution: The training results are cached in `cache/` by default.

## Training with SLURM

Training includes multiple runs with different random seeds.
To perform this execution in parallel with SLURM:
```bash
python -m seqtrack.tools.train --num_trials=5 --slurm --slurm_flags partition=devel time=1:00:00 cpus-per-task=5 gres=gpu:1
```

## Pre-trained features

To use the siamese model with pre-trained features:
```bash
python -m seqtrack.tools.train --model_params '{
        "template_size": 123,
        "search_size": 251,
        "feature_params": {
            "arch": "slim_resnet_v1_50",
            "arch_params": {
                "padding": "VALID",
                "num_blocks": 1
            }
        },
        "feature_model_file": "path/to/resnet_v1_50.ckpt"
    }'
```


# Creating experiments

To create a new experiment directory:
```bash
aux/workspace/init-experiment.sh experiments/ name
```
This will create a directory `experiments/yyyy-mm-dd-name/`.

This script requires that you have a remote called `local`.
To create this repository:
```bash
mkdir ~/projects/seqtrack.git
cd ~/projects/seqtrack.git
git init --bare

cd <repo_dir>
git remote add local ~/projects/seqtrack.git
git push local master
```


# Project structure

Here is one way to structure the project files.
```
project_root/
    seqtrack/
    dataset_cache/
    experiments/
    seqtrack.git/
```
