#!/bin/bash
activ_opts=(relu leaky_relu sigmoid tanh)
opt_opts=(adam sgd)
norm_opts=(BN LN)
dropout_opts=(0 1)
weight_decay_opts=(0 1)

run_command() {
    cuda=$1
    norm=$2
    epoch=$3
    dropout=$4
    weight_decay=$5
    opt=$6
    activ=$7
    logfile=$8

    CUDA_VISIBLE_DEVICES="$cuda" nohup python -u main.py --norm_type "$norm" --epoch "$epoch" --dropout "$dropout" --weight_decay "$weight_decay" --opt "$opt" --activation "$activ" &> "$logfile"
}

export -f run_command

for activ in "${activ_opts[@]}"; do
    for opt in "${opt_opts[@]}"; do
        for norm in "${norm_opts[@]}"; do
            run_command 0 "$norm" 40 0 0 "$opt" "$activ" "$norm-$opt-$activ-Nodropout-NoL2.log" &
            run_command 0 "$norm" 40 1 0 "$opt" "$activ" "$norm-$opt-$activ-Dropout-NoL2.log" &
            run_command 1 "$norm" 40 0 1 "$opt" "$activ" "$norm-$opt-$activ-Nodropout-L2.log" &
            run_command 1 "$norm" 40 1 1 "$opt" "$activ" "$norm-$opt-$activ-Dropout-L2.log" &
        done
    done
done

wait
