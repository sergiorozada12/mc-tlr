for method in sm dc nn fib; do
    for K in 6 12 18 24 30; do

        echo ""
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo "Sweep for method ${method} with rank ${K}"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo ""

        python3 -u main.py \
            --wandb_project "markov-chain-estimation" \
            --config_base_path "configs/sim/vary_rank/rank${K}/rank_config.yaml" \
            --config_sweep_path "configs/sim/vary_rank/rank${K}/${method}_sweep.yaml" \
            --sweep_name "varyK_rank${K}"

        echo ""
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo "Done with method ${method} with rank ${K}"
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo ""
    done
done