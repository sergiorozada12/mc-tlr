for method in fib; do
    echo "Sweep for method ${method}"
    python3 -u main.py \
        --wandb_project "markov-chain-estimation" \
        --wandb_experiment_name "sim_${method}_sweep" \
        --config_base_path "configs/sim/base_config.yaml" \
        --config_sweep_path "configs/sim/${method}_sweep.yaml"
done
