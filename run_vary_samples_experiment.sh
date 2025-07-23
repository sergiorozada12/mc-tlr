#!/bin/bash
format_time() {
    ((h=${1}/3600))
    ((m=(${1}%3600)/60))
    ((s=${1}%60))
    printf "%02d:%02d:%02d\n" $h $m $s
}

config_folder="configs/sim/vary_samples_07-05"

for method in sm nn dc fib; do
    for num_samples in 100 316 1000 3162 10000 31622 100000; do
        SECONDS=0

        echo ""
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo "Sweep for method ${method} with ${num_samples} samples"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo ""

        python3 -u main.py \
            --wandb_project "markov-chain-estimation" \
            --wandb_experiment_name "vary_samples${num_samples}_${method}" \
            --config_base_path "${config_folder}/samples${num_samples}/sim_config.yaml" \
            --config_sweep_path "${config_folder}/samples${num_samples}/${method}_sweep.yaml" \
            --sweep_name "vary_samples${num_samples}"

        echo ""
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo "Done with method ${method} with ${num_samples} samples"
        echo "Duration: $(format_time $SECONDS)"
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo ""

    done
done
