#!/bin/bash
mkdir -p checkpoints
python -u train.py --name raft-easytear --stage easytear --validation easytear --gpus 0 --num_steps 10 --batch_size 2 --lr 0.00025 --image_size 480 704 --wdecay 0.0001 --mixed_precision 