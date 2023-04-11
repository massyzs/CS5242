#!/bin/bash
for activ in relu leaky_relu 
do
  
        for layer in 1 2 3 4
        do
           
            nohup python3 -u /home/xiao/code/CS5242/CS5242/main.py --cuda 0 --norm_type LN --norm "$layer" --epoch 40 --dropout 1 --weight_decay 1 --opt adam --activation "$activ" --batch 32 &> /home/xiao/code/CS5242/CS5242/output/LN-"$layer"-adam-"$activ"-Dropout-L2.log
           
        done
        
    
done

# nohup python -u main.py --cuda 0 --norm_type BN --epoch 40 --dropout 0 --weight_decay 0 --opt adam --activation relu --aug 1 &> BN-adam-relu-Aug-Nodropout-NoL2.log