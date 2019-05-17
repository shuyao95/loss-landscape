#!/bin/bash

GPU_ID=1

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model dd_nogroup \
--model_file ../cnn/eval-DARTS_DENSE_NOGROUP-20190506-210531/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 --plot

CUDA_VISIBLE_DEVICES=1 python plot_hessian_eigen.py --x=-1:1:21 --model darts_conn1 \
--model_file ../cnn/eval-DARTS_CONN_01-20190425-104744/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --batch_size 40 --plot \
--surf_file ../cnn/eval-DARTS_CONN_01-20190425-104744/hessian.pt

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_enas \
--model_file ../cnn/eval-ENAS_NO_REG-20190507-151659/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --plot

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --x=-1:1:11 --model darts_conn1 \
--model_file ~/Downloads/eval-DARTS_CONN_01-20190425-104744//weights.pt \
--dir_type weights --xignore biasbn --xnorm filter --batch_size 1024 --plot

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model dd_node1 \
--model_file ../cnn/eval-DARTS_DENSE_CONCAT_NODE-1-20190508-143046/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 --plot

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model dd_node2 \
--model_file ../cnn/eval-DARTS_DENSE_CONCAT_NODE-2-20190509-091234/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1600 --plot

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model dd_node3 \
--model_file ../cnn/eval-DARTS_DENSE_CONCAT_NODE-3-20190509-213614/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1024 --plot

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model dd_node4 \
--model_file ../cnn/eval-DARTS_DENSE_CONCAT_NODE-4-20190511-091559/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 900 --plot

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model dd_node5 \
--model_file ../cnn/eval-DARTS_DENSE_CONCAT_NODE-5-20190511-091559/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 800 --plot