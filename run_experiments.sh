for method in fib; do
    echo "Sweep for method ${method}"
    python3 -u main.py \
        --wandb_project "markov-chain-estimation" \
        --wandb_experiment_name "taxi_${method}_sweep" \
        --config_dataset "taxi" \
        --config_base_path "configs/taxi/base_config.yaml" \
        --config_sweep_path "configs/taxi/${method}_sweep.yaml"
done
