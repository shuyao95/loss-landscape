#!/bin/bash

GPU_ID=1

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model dd_nogroup \
--model_file ../cnn/eval-DARTS_DENSE_NOGROUP-20190506-210531/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 --plot

CUDA_VISIBLE_DEVICES=1 python plot_hessian_eigen.py --x=-1:1:21 --model darts_conn1 \
--model_file ../cnn/eval-DARTS_CONN_01-20190425-104744/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --batch_size 40 --plot \
--surf_file ../cnn/eval-DARTS_CONN_01-20190425-104744/hessian.pt

#########################################################################

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_enas \
--model_file ../cnn/eval-ENAS_NO_REG-20190507-151659/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 --plot

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_enas \
--model_file ../cnn/eval-ENAS_OPS1-20190513-115141/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 --plot

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_enas \
--model_file ../cnn/eval-ENAS_OPS2-20190514-214251/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 --plot

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_enas \
--model_file ../cnn/eval-ENAS_OPS3-20190516-022843/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 --plot

#########################################################################

CUDA_VISIBLE_DEVICES=1 python plot_surface.py --x=-1:1:41 --y=-1:1:41 --model darts_amoebanet_conn1 \
--model_file ../cnn/eval-AmoebaNet_CONN1-20190515-223839/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048

CUDA_VISIBLE_DEVICES=1 python plot_surface.py --x=-1:1:41 --y=-1:1:41 --model darts_amoebanet_conn2 \
--model_file ../cnn/eval-AmoebaNet_CONN2-20190517-031723/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048

CUDA_VISIBLE_DEVICES=1 python plot_surface.py --x=-1:1:41 --y=-1:1:41 --model darts_amoebanet_conn3 \
--model_file ../cnn/eval-AmoebaNet_CONN3-20190518-074212/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048

CUDA_VISIBLE_DEVICES=1 python plot_surface.py --x=-1:1:41 --y=-1:1:41 --model darts_amoebanet_conn4 \
--model_file ../cnn/eval-AmoebaNet_CONN4-20190519-053000/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048

#########################################################################

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --x=-1:1:11 --model darts_conn1 \
--model_file ~/Downloads/eval-DARTS_CONN_01-20190425-104744//weights.pt \
--dir_type weights --xignore biasbn --xnorm filter --batch_size 1024 --plot

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:41 --y=-1:1:41 --model dd_node1 \
--model_file ../cnn/eval-DARTS_NODE1-20190515-220737/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 --plot

CUDA_VISIBLE_DEVICES=0 python plot_surface.py --x=-1:1:41 --y=-1:1:41 --model dd_node1_c46 \
--model_file ../cnn/eval-DARTS_NODE1_C46-20190518-113017/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 1548 &

CUDA_VISIBLE_DEVICES=1 python plot_surface.py --x=-1:1:41 --y=-1:1:41 --model dd_node1_c36 \
--model_file ../cnn/eval-DARTS_NODE1_C36-20190518-113017/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048

GPU_ID=4
CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:41 --y=-1:1:41 --model dd_node1 \
--model_file ../cnn/eval-DARTS_NODE1-20190515-220737/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:41 --y=-1:1:41 --model dd_node2 \
--model_file ../cnn/eval-DARTS_NODE2-20190516-161930/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:41 --y=-1:1:41 --model dd_node3 \
--model_file ../cnn/eval-DARTS_NODE3-20190515-220812/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:41 --y=-1:1:41 --model dd_node4 \
--model_file ../cnn/eval-DARTS_NODE4-20190517-001939/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_surface.py --x=-1:1:41 --y=-1:1:41 --model dd_node5 \
--model_file ../cnn/eval-DARTS_NODE5-20190515-231320/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --batch_size 2048 