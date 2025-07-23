#!/bin/bash

array_to_json() {
    local -n arr=$1
    printf -v json '[%s]' "$(IFS=,; echo "${arr[*]}")"
    echo "$json"
}

seed=1212
config_folder="configs/sim/vary_samples_07-05"
results_folder="results/results_vary_samples_07-05"
trials=100
sample_range=(100 316 1000 3162 10000 31622 100000)
dims=[5,5,5]
rank=50

mkdir -p ${config_folder}

yq -n -o=json "
    .trials = ${trials} |
    .sample_range = $(array_to_json sample_range) |
    .dims = ${dims} | 
    .rank = ${rank}
" > "${config_folder}/config.json"

###########################################
simulation configs

echo "Simulation configs"

burn_in=1000
generation_type="tensor"
generation_method="lowrank"

export GENERATION_TYPE="$generation_type"
export GENERATION_METHOD="$generation_method"

config_base_path="configs/sim/base_config.yaml"

for i in "${!sample_range[@]}"; do
    length=${sample_range[i]}

    config_new_folder="${config_folder}/samples${length}"
    mkdir -p $config_new_folder

    save_path="${results_folder}/samples${length}"
    data_dir="${results_folder}/samples${length}"
    config_new_path="${config_new_folder}/sim_config.yaml"

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

length=100000
config_new_folder="${config_folder}/samples${length}"
config_new_path="${config_new_folder}/sim_config.yaml"
python3 -u vary_samples_data.py \
    --config_base_path $config_new_path \
    --sample_range $(array_to_json sample_range) \
    --data_dir $results_folder

###########################################



############################################
# sm sweep configs

method="sm"

echo "${method} configs"
sweep_base_path="configs/sim/${method}_sweep.yaml"

K=[7,13]
for i in "${!sample_range[@]}"; do
    s=${sample_range[i]}

    sweep_new_folder="${config_folder}/samples${s}"
    sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
    export SWEEP_NAME="${method}_samples${s}"

    cp $sweep_base_path $sweep_new_path

    yq -i ".name = strenv(SWEEP_NAME)" $sweep_new_path

    yq -i ".parameters.method.parameters.${method}.parameters.K.values = ${K}" $sweep_new_path
    yq -i "del(.parameters.method.parameters.${method}.parameters.K.value)" $sweep_new_path
done
# ############################################




############################################
# nn sweep configs

method="nn"
gamma=[0.0316,0.1,0.316]
K=[7,13]
num_itrs=3000

echo "${method} configs"
sweep_base_path="configs/sim/${method}_sweep.yaml"

for i in "${!sample_range[@]}"; do 
    s=${sample_range[i]}

    sweep_new_folder="${config_folder}/samples${s}"
    sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
    export SWEEP_NAME="${method}_samples${s}"

    cp $sweep_base_path $sweep_new_path
    yq -i ".name = strenv(SWEEP_NAME)" $sweep_new_path

    vadd="value"
    vdel="values"
    yq -i ".parameters.method.parameters.${method}.parameters.num_itrs.${vadd} = ${num_itrs}" $sweep_new_path
    yq -i "del(.parameters.method.parameters.${method}.parameters.num_itrs.${vdel})" $sweep_new_path

    params=(K gamma)
    vals=($K $gamma)
    vadd="values"
    vdel="value"
    for i in "${!params[@]}"; do
        p=${params[i]}
        v=${vals[i]}
        yq -i ".parameters.method.parameters.${method}.parameters.${p}.${vadd} = ${v}" $sweep_new_path
        yq -i "del(.parameters.method.parameters.${method}.parameters.${p}.${vdel})" $sweep_new_path
    done
done
############################################



############################################
# dc sweep configs

method="dc"
K=[7,13]
num_itrs=3000
alpha=[0.1,0.178,0.316]
gamma=[0.1]
num_inn_itrs=10

echo "${method} configs"
sweep_base_path="configs/sim/${method}_sweep.yaml"
sweep_new_path="${config_folder}/${method}_sweep.yaml"

for i in "${!sample_range[@]}"; do 
    s=${sample_range[i]}

    sweep_new_folder="${config_folder}/samples${s}"
    sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
    export SWEEP_NAME="${method}_samples${s}"

    cp $sweep_base_path $sweep_new_path
    yq -i ".name = strenv(SWEEP_NAME)" $sweep_new_path

    params=(num_itrs num_inn_itrs)
    vals=($num_itrs $num_inn_itrs)
    vadd="value"
    vdel="values"
    for i in "${!params[@]}"; do
        p=${params[i]}
        v=${vals[i]}
        yq -i ".parameters.method.parameters.${method}.parameters.${p}.${vadd} = ${v}" $sweep_new_path
        yq -i "del(.parameters.method.parameters.${method}.parameters.${p}.${vdel})" $sweep_new_path
    done

    params=(K alpha gamma)
    vals=($K $alpha $gamma)
    vadd="values"
    vdel="value"
    for i in "${!params[@]}"; do
        p=${params[i]}
        v=${vals[i]}
        yq -i ".parameters.method.parameters.${method}.parameters.${p}.${vadd} = ${v}" $sweep_new_path
        yq -i "del(.parameters.method.parameters.${method}.parameters.${p}.${vdel})" $sweep_new_path
    done
done
############################################



############################################
# fib sweep configs

method="fib"
K=[50,100]
alpha_factor=[10.0,31.6,100.0,316.2]
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
sweep_new_path="${config_folder}/${method}_sweep.yaml"

for i in "${!sample_range[@]}"; do 
    s=${sample_range[i]}

    sweep_new_folder="${config_folder}/samples${s}"
    sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
    export SWEEP_NAME="${method}_samples${s}"

    cp $sweep_base_path $sweep_new_path
    yq -i ".name = strenv(SWEEP_NAME)" $sweep_new_path

    params=(alpha_weight gamma_factor B B_max increase_B num_itrs forget_window tol)
    vals=($alpha_weight $gamma_factor $B $B_max $increase_B $num_itrs $forget_window $tol)
    vadd="value"
    vdel="values"
    for i in "${!params[@]}"; do
        p=${params[i]}
        v=${vals[i]}
        yq -i ".parameters.method.parameters.${method}.parameters.${p}.${vadd} = ${v}" $sweep_new_path
        yq -i "del(.parameters.method.parameters.${method}.parameters.${p}.${vdel})" $sweep_new_path
    done

    params=(K alpha_factor gamma_weight beta)
    vals=($K $alpha_factor $gamma_weight $beta)
    vadd="values"
    vdel="value"
    for i in "${!params[@]}"; do
        p=${params[i]}
        v=${vals[i]}
        yq -i ".parameters.method.parameters.${method}.parameters.${p}.${vadd} = ${v}" $sweep_new_path
        yq -i "del(.parameters.method.parameters.${method}.parameters.${p}.${vdel})" $sweep_new_path
    done
done
############################################

echo "Done with configs"