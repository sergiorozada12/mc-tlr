for method in dc nn sm fib; do
    echo "Sweep for method ${method}"
    python3 -u sweep.py \
        --experiment_name "${method}_sweep" \
        --data_path "data/sim_sweep.npy" \
        --sweep_config "sweeps/${method}_sweep.yaml"
done
