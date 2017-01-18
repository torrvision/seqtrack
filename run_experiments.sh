#!/bin/sh

echo $(hostname)

export PYTHONUNBUFFERED="TRUE"

LOG="logs/`date +'%Y-%m-%d_%H-%M-%S'`.txt"
echo $LOG

python main.py $@ 2>&1 | tee -a $LOG 
