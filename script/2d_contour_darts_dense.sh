#!/bin/bash

GPU_ID=1

# darts v2 without any regularization
CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_v2 \
--model_file ../cnn/eval-DARTS_NO-REG-20190425-104947/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024

# darts v2 with cutout
CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_v2 \
--model_file ../cnn/eval-DARTS_ONLY_CUTOUT-20190429-104935/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --plot

# darts v2 with cutout and auxiliary
CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_v2 \
--model_file ../cnn/eval-DARTS_NO_DROP_PATH-20190429-104935/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --auxiliary --plot

# darts v2 with cutout, auxiliary and drop path
CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_v2 \
--model_file ../cnn/eval-DARTS-20190507-101323/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --auxiliary