#!/bin/bash

# ===========================================================
# 2d loss contours for ResNet-56
# ===========================================================

CUDA_VISIBLE_DEVICES=1 python plot_surface.py --x=-1:1:51 --y=-1:1:51 --model darts_v2 \
--model_file ../cnn/eval-DARTS_NO-REG-20190425-104947/weights.pt \
--cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter