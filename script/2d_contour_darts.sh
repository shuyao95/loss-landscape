#!/bin/bash

# ===========================================================
# 2d loss contours for darts
# ===========================================================
GPU_ID=1

# darts v2 without any regularization
CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_v2 \
--model_file ../cnn/eval-DARTS_NO-REG-20190425-104947/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024

CUDA_VISIBLE_DEVICES=5 python train_dd.py --batch_size 60 --drop_path_prob 0 --arch concat --save DARTS_DENSE_CONCAT_NODE-4 --seed 0 --nodes 4 --init_channels 24
CUDA_VISIBLE_DEVICES=5 python train_dd.py --batch_size 60 --drop_path_prob 0 --arch concat --save TEST --seed 0 --nodes 4 --init_channels 24

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

GPU_ID=1
# darts ops with cutout, auxiliary and drop path
CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_ops1 \
--model_file ../cnn/eval-DARTS_OPS_01-20190410-171722/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --auxiliary

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_ops2 \
--model_file ../cnn/eval-DARTS_OPS_02-20190412-061658/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --auxiliary

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_ops3 \
--model_file ../cnn/eval-DARTS_OPS_03-20190413-132935/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --auxiliary

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_ops4 \
--model_file ../cnn/eval-DARTS_OPS_04-20190414-173544/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --auxiliary

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_ops5 \
--model_file ../cnn/eval-DARTS_OPS_05-20190416-013321/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --auxiliary

GPU_ID=0
# darts conn with cutout, auxiliary and drop path
CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_conn1 \
--model_file ../cnn/eval-DARTS_CONN_01-20190417-103208/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --auxiliary

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_conn2 \
--model_file ../cnn/eval-DARTS_CONN_02-20190418-225844/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --auxiliary

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_conn3 \
--model_file ../cnn/eval-DARTS_CONN_03-20190420-122602/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --auxiliary

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_conn4 \
--model_file ../cnn/eval-DARTS_CONN_04-20190421-224840/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --auxiliary


GPU_ID=1
# darts ops without cutout, auxiliary and drop path
CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_ops1 \
--model_file ../cnn/eval-DARTS_OPS_01-20190425-104500/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_ops2 \
--model_file ../cnn/eval-DARTS_OPS_02-20190429-104257/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_ops3 \
--model_file ../cnn/eval-DARTS_OPS_03-20190430-114142/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_ops4 \
--model_file ../cnn/eval-DARTS_OPS_04-20190501-095558/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_ops5 \
--model_file ../cnn/eval-DARTS_OPS_05-20190502-113742/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024

GPU_ID=0
# darts conn without cutout, auxiliary and drop path
CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_conn1 \
--model_file ../cnn/eval-DARTS_CONN_01-20190425-104744/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_conn2 \
--model_file ../cnn/eval-DARTS_CONN_02-20190502-103715/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_conn3 \
--model_file ../cnn/eval-DARTS_CONN_03-20190503-234341/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_conn4 \
--model_file ../cnn/eval-DARTS_CONN_04-20190502-103748/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024
