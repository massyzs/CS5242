#!/bin/bash
for activ in relu leaky_relu sigmoid tanh
do
    for opt in adam sgd
    do
        for norm in BN LN
        do
            nohup python -u main.py --cuda 0 --norm_type "$norm" --epoch 40 --dropout 0 --weight_decay 0 --opt "$opt" --activation "$activ" &> "$norm"-"$opt"-"$activ"-Nodropout-NoL2.log
            nohup python -u main.py --cuda 0 --norm_type "$norm" --epoch 40 --dropout 1 --weight_decay 0 --opt "$opt" --activation "$activ" &> "$norm"-"$opt"-"$activ"-Dropout-NoL2.log
            nohup python -u main.py --cuda 1 --norm_type "$norm" --epoch 40 --dropout 0 --weight_decay 1 --opt "$opt" --activation "$activ" &> "$norm"-"$opt"-"$activ"-Nodropout-L2.log
            nohup python -u main.py --cuda 1 --norm_type "$norm" --epoch 40 --dropout 1 --weight_decay 1 --opt "$opt" --activation "$activ" &> "$norm"-"$opt"-"$activ"-Dropout-L2.log
        done
    done
done