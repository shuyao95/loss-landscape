#!/bin/bash

GPU_ID=1

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model dd_nogroup \
--model_file ../cnn/eval-DARTS_DENSE_NOGROUP-20190506-210531/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 --plot

CUDA_VISIBLE_DEVICES=0 python plot_hessian_eigen.py --x=-1:1:51 --y=-1:1:51 --model darts_ops1 \
--model_file ../cnn/eval-DARTS_CONN_01-20190417-103208/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 --plot

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model dd_node1 \
--model_file ../cnn/eval-DARTS_DENSE_CONCAT_NODE-1-20190508-143046/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 --plot

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model dd_node1 \
--model_file ../cnn/eval-DARTS_DENSE_CONCAT_NODE-2-20190509-091234/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 --plot
