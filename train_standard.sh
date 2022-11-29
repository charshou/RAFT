#!/bin/bash
mkdir -p checkpoints
python -u train.py --name raft-easytear --stage easytear --validation easytear --gpus 0 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 480 703 --wdecay 0.0001