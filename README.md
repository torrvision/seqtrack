# Visual object tracking using rnn 
Namhoon Lee | Torr Vision Group, the University of Oxford


# Usage:
To train a model simply pass train mode flag:
> $ python --mode=train --dataset=some_dataset

In order to test, you need to give the path to a trained model as well as test mode flag:
> $ python --mode=test --restore --restore_model=path_to_model_file

For more information on the available options, please refer to scripts.


# Note:
- (flag) debugmode: run with decreased iteration numbers for quick check 
- (flag) nosave: if you don't to save your training results (used for debugging)
- (command) CUDA_VISIBLE_DEVICES=gpu_number: 


# To run experiments in torrnode (using Jack's script)

Create local git repository
```
> mkdir ~/projects/rnntracking_dev.git
> cd ~/projects/rnntracking_dev.git
> git init --bare

> cd <source directory>
> git remote add local /your_path/rnntracking_dev.git
> git push local master
```

Running
```
Enable virtual environment (assuming one global virtual environment)
> source ~/virenv/bin/activate

Install necessary python packages (if it's your first time running experiments)
> pip install opencv-python, scipy, pillow, xmltodict, progressbar, matplotlib, h5py plus tensorflow
```
