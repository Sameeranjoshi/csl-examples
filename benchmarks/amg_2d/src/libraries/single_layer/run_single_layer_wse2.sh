#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
# This ensures the script stops on any error rather than continuing
set -e
# Instruction Trace log
# export SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace
# make a directory for the output out if it doesn't exist
mkdir -p out

# Array of configurations
# Format: [A_rows, A_cols, R_rows, R_cols, kernel_rows, kernel_cols, fabric_dim_width, fabric_dim_height, out_name]
declare -A configs=(    # don't scale beyond 32 PEs, crashes.
    # expriments
    # ["config1"]="4 4 4 4 2 2 9 4 out/out_A4_4_R4_4_PE2_2"
    # ["config2"]="4 4 4 4 4 4 11 6 out/out_A4_4_R4_4_PE4_4"

    # ["config3"]="16 16 16 16 2 2 9 4 out/out_A16_16_R16_16_PE2_2"
    # ["config4"]="16 16 16 16 4 4 11 6 out/out_A16_16_R16_16_PE4_4"
    # ["config5"]="16 16 16 16 8 8 15 10 out/out_A16_16_R16_16_PE8_8"

    # ["config6"]="64 64 64 64 2 2 9 4 out/out_A64_64_R64_64_PE2_2"
    # ["config7"]="64 64 64 64 4 4 11 6 out/out_A64_64_R64_64_PE4_4"
    # ["config8"]="64 64 64 64 8 8 15 10 out/out_A64_64_R64_64_PE8_8"
    # ["config9"]="64 64 64 64 16 16 23 18 out/out_A64_64_R64_64_PE16_16"
    # ["config10"]="64 64 64 64 32 32 39 34 out/out_A64_64_R64_64_PE32_32"

    # ["config11"]="128 128 128 128 4 4 11 6 out/out_A128_128_R128_128_PE4_4"
    # ["config12"]="128 128 128 128 8 8 15 10 out/out_A128_128_R128_128_PE8_8"

    # ["config13"]="256 256 256 256 8 8 15 10 out/out_A256_256_R256_256_PE8_8"

)
# declare -A configs=(
#     # ["config1"]="256 256 256 256 64 64 71 66 out/out_A256_256_R256_256_PE64_64" # crashed
#     # ["config3"]="128 128 128 128 64 64 71 66 out/out_A128_128_R128_128_PE64_64" # crashed
# )


# Array of uneven configurations
# Format: [A_rows, A_cols, R_rows, R_cols, kernel_rows, kernel_cols, fabric_dim_width, fabric_dim_height, out_name]
# don't scale beyond 32 PEs, crashes.
declare -A configs_pad=(

    # input-square, PE-square. - Passes
    # PE odd.
    ["config1"]="10 10 10 10 5 5 12 7 out/out_A10_10_R5_10_PE5_5"   # no padding
    ["config2"]="11 11 11 11 5 5 12 7 out/out_A11_11_R5_11_PE5_5"   # pad 
    ["config3"]="3 3 3 3 5 5 17 12 out/out_A3_3_R3_3_PE5_5" # less than PE(pad)
    # PE even.
    ["config4"]="10 10 10 10 4 4 12 7 out/out_A10_10_R4_10_PE4_4"
    ["config5"]="9 9 9 9 4 4 12 7 out/out_A9_9_R4_9_PE4_4"
    ["config6"]="11 11 11 11 4 4 12 7 out/out_A11_11_R4_11_PE4_4"

    # input - non-square, PE - square.(should fail)
    # ["config4"]="11 12 5 11 5 5 12 7 out/out_A11_12_R5_11_PE5_5"
    # input - square, PE - non-square.(should fail)
    # ["config5"]="5 5 5 5 2 3 13 8 out/out_A5_5_R5_5_PE2_3"

)

get_padded_dims() {
    local matrix_rows=$1
    local matrix_cols=$2
    local kernel_rows=$3
    local kernel_cols=$4

    local pad_rows=$(( (kernel_rows - (matrix_rows % kernel_rows)) % kernel_rows ))
    local pad_cols=$(( (kernel_cols - (matrix_cols % kernel_cols)) % kernel_cols ))
    local padded_rows=$((matrix_rows + pad_rows))
    local padded_cols=$((matrix_cols + pad_cols))

    echo "$padded_rows $padded_cols"
}

# Function to run a single configuration
run_config() {
    local A_rows=$1
    local A_cols=$2
    local R_rows=$3
    local R_cols=$4
    local kernel_rows=$5
    local kernel_cols=$6
    local fabric_width=$7
    local fabric_height=$8
    local out_name=$9

    echo "Running configuration: $out_name"
    echo "Fabric dimensions: ${fabric_width}x${fabric_height}"
    
    # Calculate fabric offsets (centered)
    local offset_x=4
    local offset_y=1


    # Pad data before sending to PE.
    # Pad dimensions for A
    local padded_A_dims=$(get_padded_dims "$A_rows" "$A_cols" "$kernel_rows" "$kernel_cols")
    local padded_A_rows=$(echo "$padded_A_dims" | awk '{print $1}')
    local padded_A_cols=$(echo "$padded_A_dims" | awk '{print $2}')
    echo "Padding A from ${A_rows}x${A_cols} to ${padded_A_rows}x${padded_A_cols}"

    # Pad dimensions for R
    local padded_R_dims=$(get_padded_dims "$R_rows" "$R_cols" "$kernel_rows" "$kernel_cols")
    local padded_R_rows=$(echo "$padded_R_dims" | awk '{print $1}')
    local padded_R_cols=$(echo "$padded_R_dims" | awk '{print $2}')
    echo "Padding R from ${R_rows}x${R_cols} to ${padded_R_rows}x${padded_R_cols}"

    # Run CSLC command
    cslc --arch=wse2 ./single_layer_layout.csl \
        --fabric-dims=${fabric_width},${fabric_height} \
        --fabric-offsets=${offset_x},${offset_y} \
        --params=layer_rows_A:${padded_A_rows},layer_cols_A:${padded_A_cols} \
        --params=layer_rows_R:${padded_R_rows},layer_cols_R:${padded_R_cols} \
        --params=layer_start_x:0,layer_start_y:0,layer_kernel_rows:${kernel_rows},layer_kernel_cols:${kernel_cols},layer_index:0 \
        --memcpy --channels=1 --max-inlined-iterations=1000000 -o ${out_name}

    # Run Python script
    # TODO: Use .json file as params in run.py
    cs_python single_layer_run.py --name ${out_name} --A_rows ${A_rows} --A_cols ${A_cols} --R_rows ${R_rows} --R_cols ${R_cols}
    # cs_python test_memcpy.py --M ${A_rows} --N ${A_cols} --kernel_rows ${kernel_rows} --kernel_cols ${kernel_cols} --name ${out_name}
    
    echo "Completed configuration: $out_name"
    echo "Moving sim* files to out directory"
    rm -rf ${out_name}/simfab_traces
    mv --force sim* ${out_name}
    echo "----------------------------------------"
}

echo "Running single layer experiments"
# Get the maximum config number dynamically
max_config=$(for key in "${!configs[@]}"; do echo "${key#config}"; done | sort -n | tail -n1)

# Run all configurations in order from config1 to the last config
for i in $(seq 1 $max_config); do
    config="config$i"
    if [[ -n "${configs[$config]}" ]]; then
        read -r A_rows A_cols R_rows R_cols kernel_rows kernel_cols fabric_width fabric_height out_name <<< "${configs[$config]}"
        run_config "$A_rows" "$A_cols" "$R_rows" "$R_cols" "$kernel_rows" "$kernel_cols" "$fabric_width" "$fabric_height" "$out_name"
    fi
done

echo "Running padding experiments"
# Test padding
max_config_pad=$(for key in "${!configs_pad[@]}"; do echo "${key#config}"; done | sort -n | tail -n1)
# Run all configurations in order from config1 to the last config
for i in $(seq 1 $max_config_pad); do
    config="config$i"
    if [[ -n "${configs_pad[$config]}" ]]; then
        read -r A_rows A_cols R_rows R_cols kernel_rows kernel_cols fabric_width fabric_height out_name <<< "${configs_pad[$config]}"
        run_config "$A_rows" "$A_cols" "$R_rows" "$R_cols" "$kernel_rows" "$kernel_cols" "$fabric_width" "$fabric_height" "$out_name"
    fi
done

# Optional: Run a specific configuration by uncommenting and modifying the line below
# run_config 128 128 128 128 4 4 11 6 out_A128_128_R128_128_PE4_4
