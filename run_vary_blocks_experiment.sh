#!/bin/bash
format_time() {
    ((h=${1}/3600))
    ((m=(${1}%3600)/60))
    ((s=${1}%60))
    printf "%02d:%02d:%02d\n" $h $m $s
}

config_folder="configs/sim/vary_blocks_07-01-2200"

for method in sm nn dc fib; do
    for num_blocks in 2 4 6 8 10 12 14; do
        SECONDS=0

        echo ""
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo "Sweep for method ${method} with ${num_blocks} blocks"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo ""

        python3 -u main.py \
            --wandb_project "markov-chain-estimation" \
            --wandb_experiment_name "vary_blocks${num_blocks}_${method}" \
            --config_base_path "${config_folder}/blocks${num_blocks}/sim_config.yaml" \
            --config_sweep_path "${config_folder}/blocks${num_blocks}/${method}_sweep.yaml" \
            --sweep_name "vary_blocks${num_blocks}"

        echo ""
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo "Done with method ${method} with ${num_blocks} blocks"
        echo "Duration: $(format_time $SECONDS)"
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo ""
    done
done
