#!/bin/bash

array_to_json() {
    local -n arr=$1
    printf -v json '[%s]' "$(IFS=,; echo "${arr[*]}")"
    echo "$json"
}



############################################

seed=1111
config_folder="configs/sim/single_dim_07-20-0740"
results_folder="results/results_single_dim_07-20-0740"
trials=10
rank=35
Kten=35
Kmat=4
dims=[2,10,10]
D=3
length=1000

mkdir -p $config_folder

yq -n -o=json "
    .trials = ${trials} |
    .rank = ${rank} |
    .dims = ${dims} |
    .length = ${length} | 
    .Kmat = ${Kmat} |
    .Kten = ${Kten}
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

config_new_folder="${config_folder}/marg"
mkdir -p $config_new_folder

save_path="${results_folder}"
data_dir="${results_folder}"
config_new_path="${config_new_folder}/sim_config.yaml"

export SAVE_PATH="$save_path"
export DATA_DIR="${data_dir}/data_marg"

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

python3 -u single_dim_data.py \
    --config_base_path $config_new_path \
    --data_dir $results_folder

for ((d = 0; d < $D ; d++)); do
    mkdir -p "${config_folder}/single${d}"
    config_single_path="${config_folder}/single${d}/sim_config.yaml"
    export DATA_DIR_SINGLE="${results_folder}/data_single${d}"
    yq ".dataset.data_dir = strenv(DATA_DIR_SINGLE)" $config_new_path > $config_single_path
done
############################################



############################################
# sm sweep configs

method="sm"

echo "${method} configs"
sweep_base_path="configs/sim/${method}_sweep.yaml"

sim_set=(marg single0 single1 single2)
K_range=($Kmat $Kten $Kten $Kten)

for i in "${!sim_set[@]}"; do
    sim=${sim_set[i]}
    K=${K_range[i]}

    sweep_new_folder="${config_folder}/${sim}"
    sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
    sweep_name="${method}_${sim}"
    export SWEEP_NAME="$sweep_name"

    mkdir -p $sweep_new_folder
    cp $sweep_base_path $sweep_new_path

    yq -i " .name = strenv(SWEEP_NAME) " $sweep_new_path

    yq -i " .parameters.method.parameters.${method}.parameters.K.value = ${K} " $sweep_new_path
    yq -i "del(.parameters.method.parameters.${method}.parameters.K.values)" $sweep_new_path
done
############################################



############################################
# nn sweep configs

method="nn"
gamma=[0.03,0.3,0.5]
num_itrs=1000

echo "${method} configs"
sweep_base_path="configs/sim/${method}_sweep.yaml"

sim_set=(marg single0 single1 single2)
K_range=($Kmat $Kten $Kten $Kten)

for i in "${!sim_set[@]}"; do
    sim=${sim_set[i]}
    K=${K_range[i]}

    sweep_new_folder="${config_folder}/${sim}"
    sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
    sweep_name="${method}_${sim}"
    export SWEEP_NAME="$sweep_name"

    mkdir -p $sweep_new_folder
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

method="dc"
alpha=[0.1,0.316,1.0]
gamma=[0.1,0.316]
num_itrs=3000
num_inn_itrs=10

echo "${method} configs"
sweep_base_path="configs/sim/${method}_sweep.yaml"

sim_set=(marg single0 single1 single2)
K_range=($Kmat $Kten $Kten $Kten)

for i in "${!sim_set[@]}"; do
    sim=${sim_set[i]}
    K=${K_range[i]}

    sweep_new_folder="${config_folder}/${sim}"
    sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
    sweep_name="${method}_${sim}"
    export SWEEP_NAME="$sweep_name"

    mkdir -p $sweep_new_folder
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

    vadd="values"
    vdel="value"
    yq -i ".parameters.method.parameters.${method}.parameters.alpha.${vadd} = ${alpha}" $sweep_new_path
    yq -i "del(.parameters.method.parameters.${method}.parameters.alpha.${vdel})" $sweep_new_path
    yq -i ".parameters.method.parameters.${method}.parameters.gamma.${vadd} = ${gamma}" $sweep_new_path
    yq -i "del(.parameters.method.parameters.${method}.parameters.gamma.${vdel})" $sweep_new_path
done
############################################



############################################
# fib sweep configs

method="fib"
alpha_factor=[1.0,3.0,10.0,30.0,100.0]
# alpha_factor=[10.0,100.0,200.0]
alpha_weight=1.0
gamma_factor=0.0
gamma_weight=[0.0,0.01]
beta=[0.3,0.5]
B=200
B_max=1000
increase_B=True
num_itrs=150000
forget_window=5000
tol=1e-10

echo "${method} configs"
sweep_base_path="configs/sim/${method}_sweep.yaml"

K=$Kten
sim="marg"
alpha_factor=[1.0,3.0,10.0,30.0,100.0]
# alpha_factor=[1.0,3.0,5.0]

sweep_new_folder="${config_folder}/${sim}"
sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
sweep_name="${method}_${sim}"
export SWEEP_NAME="$sweep_name"

mkdir -p $sweep_new_folder
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


sim_set=(single0 single1 single2)
K_range=($Kten $Kten $Kten)
# K=$Kten
# sim="single"
# alpha_factor=[1.0,3.0,10.0]
alpha_factor=[0.1,0.3,1.0,3.0,10.0]

for i in "${!sim_set[@]}"; do
    sim=${sim_set[i]}
    K=${K_range[i]}

    sweep_new_folder="${config_folder}/${sim}"
    sweep_new_path="${sweep_new_folder}/${method}_sweep.yaml"
    sweep_name="${method}_${sim}"
    export SWEEP_NAME="$sweep_name"

    mkdir -p $sweep_new_folder
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
