#!/bin/bash
format_time() {
    ((h=${1}/3600))
    ((m=(${1}%3600)/60))
    ((s=${1}%60))
    printf "%02d:%02d:%02d\n" $h $m $s
}

config_folder="configs/sim/single_dim_07-20-0740"

for method in sm nn dc fib; do
    SECONDS=0
    sim_setting="marg"

    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "Sweep for method ${method} with ${sim_setting} trajectory"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ""

    python3 -u main.py \
        --wandb_project "markov-chain-estimation" \
        --wandb_experiment_name "single_dim_${sim_setting}_${method}" \
        --config_base_path "${config_folder}/${sim_setting}/sim_config.yaml" \
        --config_sweep_path "${config_folder}/${sim_setting}/${method}_sweep.yaml" \
        --sweep_name "single_dim_${sim_setting}"

    echo ""
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo "Done with method ${method} with ${sim_setting} trajectory"
    echo "Duration: $(format_time $SECONDS)"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo ""
    
    
    SECONDS=0
    for ((d = 0; d < 3; d++)); do
        sim_setting="single${d}"

        echo ""
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo "Sweep for method ${method} with ${sim_setting} trajectory"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo ""
        
        python3 -u main.py \
            --wandb_project "markov-chain-estimation" \
            --wandb_experiment_name "single_dim_${sim_setting}_${method}" \
            --config_base_path "${config_folder}/${sim_setting}/sim_config.yaml" \
            --config_sweep_path "${config_folder}/${sim_setting}/${method}_sweep.yaml" \
            --sweep_name "single_dim_${sim_setting}"
        
        echo ""
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo "Done with method ${method} with ${sim_setting} trajectory"
        echo "Duration: $(format_time $SECONDS)"
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo ""
    done
done
