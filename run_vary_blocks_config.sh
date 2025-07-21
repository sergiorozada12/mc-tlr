#!/bin/bash

array_to_json() {
    local -n arr=$1
    printf -v json '[%s]' "$(IFS=,; echo "${arr[*]}")"
    echo "$json"
}

seed=1111
config_folder="configs/sim/vary_blocks_07-01-2200"
results_folder="results/results_vary_blocks_07-01-2200"
block_range=(2 4 6 8 10 12 14)
sample_range=(1109 2661 4381 6210 8120 10094 12122)
Kmat_range=(17 10 8 7 6 6 6)
rank=50
Kten=$rank
trials=100
dim_base=4,4

yq -n -o=json "
    .trials = ${trials} | 
    .block_range = $(array_to_json block_range) | 
    .rank = ${rank} | 
    .dim_base = [${dim_base}] | 
    .matrix_ranks = $(array_to_json Kmat_range) | 
    .lengths = $(array_to_json sample_range)
" > "${config_folder}/config.json"

############################################
# simulation configs

echo "Simulation configs"

burn_in=1000
generation_type="tensor"
generation_method="block"

config_base_path="configs/sim/base_config.yaml"

export GENERATION_TYPE="$generation_type"
export GENERATION_METHOD="$generation_method"

for i in "${!block_range[@]}"; do
    b=${block_range[i]}
    length=${sample_range[i]}

    config_new_folder="${config_folder}/blocks${b}"
    mkdir -p $config_new_folder

    save_path="${results_folder}/results_blocks${b}"
    data_dir="${results_folder}/results_blocks${b}"
    config_new_path="${config_new_folder}/sim_config.yaml"
    dims=[${b},${dim_base}]

    export SAVE_PATH="$save_path"
    export DATA_DIR="$data_dir"

    yq "
        .general.seed = ${seed} | 
        .general.save_path = strenv(SAVE_PATH) | 
        .dataset.dims = ${dims} | 
        .dataset.data_dir = strenv(DATA_DIR) | 
        .dataset.rank = ${rank} | 
        .dataset.burn_in = ${burn_in} | 
        .dataset.length = ${length} | 
        .dataset.trials = ${trials} | 
        .dataset.generation_type = strenv(GENERATION_TYPE) | 
        .dataset.generation_method = strenv(GENERATION_METHOD)
    " $config_base_path > $config_new_path
done
############################################



############################################
# sm sweep configs

# Kmat_range=(7 5 4 4 3 3 3 3 3)
# Kmat_range=(14 11 9 8 8 7 7)

method="sm"

echo "${method} configs"
sweep_base_path="configs/sim/${method}_sweep.yaml"

for i in "${!block_range[@]}"; do
    b=${block_range[i]}
    K=${Kmat_range[i]}

    sweep_new_folder="${config_folder}/blocks${b}"
    sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
    sweep_name="${method}_blocks${b}"
    export SWEEP_NAME="$sweep_name"

    cp $sweep_base_path $sweep_new_path

    yq -i " .name = strenv(SWEEP_NAME) " $sweep_new_path

    yq -i " .parameters.method.parameters.${method}.parameters.K.value = ${K} " $sweep_new_path
    yq -i "del(.parameters.method.parameters.${method}.parameters.K.values)" $sweep_new_path
done
############################################



############################################
# nn sweep configs

# Kmat_range=(7 5 4 4 3 3 3 3 3)
# Kmat_range=(14 11 9 8 8 7 7)
# gamma=[0.0316,0.1,0.316]

method="nn"
gamma=[0.0316,0.0562,0.1,0.178,0.316]
num_itrs=3000

echo "${method} configs"
sweep_base_path="configs/sim/${method}_sweep.yaml"

for i in "${!block_range[@]}"; do
    b=${block_range[i]}
    K=${Kmat_range[i]}

    sweep_new_folder="${config_folder}/blocks${b}"
    sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
    sweep_name="${method}_blocks${b}"
    export SWEEP_NAME="$sweep_name"

    cp $sweep_base_path $sweep_new_path
    yq -i " .name = strenv(SWEEP_NAME) " $sweep_new_path

    params=(K num_itrs)
    vals=($K $num_itrs)
    vadd="value"
    vdel="values"
    for i in "${!params[@]}"; do
        p=${params[i]}
        v=${vals[i]}
        yq -i ".parameters.method.parameters.${method}.parameters.${p}.${vadd} = ${v}" $sweep_new_path
        yq -i "del(.parameters.method.parameters.${method}.parameters.${p}.${vdel})" $sweep_new_path
    done

    vadd="values"
    vdel="value"
    yq -i ".parameters.method.parameters.${method}.parameters.gamma.${vadd} = ${gamma}" $sweep_new_path
    yq -i "del(.parameters.method.parameters.${method}.parameters.gamma.${vdel})" $sweep_new_path
done
############################################



############################################
# dc sweep configs

# Kmat_range=(7 5 4 4 3 3 3 3 3)
# Kmat_range=(14 11 9 8 8 7 7)

method="dc"
alpha=[0.1,0.178,0.316]
gamma=[0.1,0.178,0.316]
num_itrs=3000
num_inn_itrs=10

echo "${method} configs"
sweep_base_path="configs/sim/${method}_sweep.yaml"

for i in "${!block_range[@]}"; do
    b=${block_range[i]}
    K=${Kmat_range[i]}

    sweep_new_folder="${config_folder}/blocks${b}"
    sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
    sweep_name="${method}_blocks${b}"
    export SWEEP_NAME="$sweep_name"

    cp $sweep_base_path $sweep_new_path
    yq -i " .name = strenv(SWEEP_NAME) " $sweep_new_path

    params=(K num_itrs num_inn_itrs)
    vals=($K $num_itrs $num_inn_itrs)
    vadd="value"
    vdel="values"
    for i in "${!params[@]}";do
        p=${params[i]}
        v=${vals[i]}
        yq -i ".parameters.method.parameters.${method}.parameters.${p}.${vadd} = ${v}" $sweep_new_path
        yq -i "del(.parameters.method.parameters.${method}.parameters.${p}.${vdel})" $sweep_new_path
    done

    params=(alpha gamma)
    vals=($alpha $gamma)
    vadd="values"
    vdel="value"
    for i in "${!params[@]}";do
        p=${params[i]}
        v=${vals[i]}
        yq -i ".parameters.method.parameters.${method}.parameters.${p}.${vadd} = ${v}" $sweep_new_path
        yq -i "del(.parameters.method.parameters.${method}.parameters.${p}.${vdel})" $sweep_new_path
    done
done
############################################



############################################
# fib sweep configs

# Larger alpha_factor, alpha_weight=1. fine.
# gamma_factor=0 best. gamma_weight=0.01 can help. consider gamma_factor=0.001 (alpha_factor too large)
# beta=.3 better than beta=.1. Consider beta=.5
# Higher B=100 better, higher B_max=500 worse (when alpha_factor too small).

method="fib"
alpha_factor=[100.0,171.0,292.4,500.0]
alpha_weight=1.0
gamma_factor=0.0
gamma_weight=[0.0,0.01]
beta=[0.3,0.5]
B=200
B_max=1000
increase_B=False
num_itrs=150000
forget_window=5000
tol=1e-10

echo "${method} configs"
sweep_base_path="configs/sim/${method}_sweep.yaml"

for i in "${!block_range[@]}"; do
    b=${block_range[i]}

    sweep_new_folder="${config_folder}/blocks${b}"
    sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
    sweep_name="${method}_blocks${b}"
    export SWEEP_NAME="$sweep_name"

    cp $sweep_base_path $sweep_new_path
    yq -i " .name = strenv(SWEEP_NAME) " $sweep_new_path

    params=(K alpha_weight gamma_factor B B_max increase_B num_itrs forget_window tol)
    vals=($Kten $alpha_weight $gamma_factor $B $B_max $increase_B $num_itrs $forget_window $tol)
    vadd="value"
    vdel="values"
    for i in "${!params[@]}";do
        p=${params[i]}
        v=${vals[i]}
        yq -i ".parameters.method.parameters.${method}.parameters.${p}.${vadd} = ${v}" $sweep_new_path
        yq -i "del(.parameters.method.parameters.${method}.parameters.${p}.${vdel})" $sweep_new_path
    done

    params=(alpha_factor gamma_weight beta)
    vals=($alpha_factor $gamma_weight $beta)
    vadd="values"
    vdel="value"
    for i in "${!params[@]}";do
        p=${params[i]}
        v=${vals[i]}
        yq -i ".parameters.method.parameters.${method}.parameters.${p}.${vadd} = ${v}" $sweep_new_path
        yq -i "del(.parameters.method.parameters.${method}.parameters.${p}.${vdel})" $sweep_new_path
    done
done
# ############################################

echo "Done with configs"