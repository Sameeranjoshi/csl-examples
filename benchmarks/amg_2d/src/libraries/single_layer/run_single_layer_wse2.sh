#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
# This ensures the script stops on any error rather than continuing
set -e

# make a directory for the output out if it doesn't exist
mkdir -p out

# Array of configurations
# Format: [A_rows, A_cols, R_rows, R_cols, kernel_rows, kernel_cols, fabric_dim_width, fabric_dim_height, out_name]
# declare -A configs=(
#     # ["config1"]="256 256 256 256 64 64 71 66 out/out_A256_256_R256_256_PE64_64" # crashed
#     # ["config2"]="256 256 256 256 8 8 15 10 out/out_A256_256_R256_256_PE8_8"     

#     # ["config3"]="128 128 128 128 64 64 71 66 out/out_A128_128_R128_128_PE64_64" # crashed
#     # ["config4"]="128 128 128 128 8 8 15 10 out/out_A128_128_R128_128_PE8_8"

#     ["config5"]="64 64 64 64 32 32 39 34 out/out_A64_64_R64_64_PE32_32"
#     ["config6"]="64 64 64 64 16 16 23 18 out/out_A64_64_R64_64_PE16_16"
#     ["config7"]="64 64 64 64 8 8 15 10 out/out_A64_64_R64_64_PE8_8"
#     ["config8"]="64 64 64 64 4 4 11 6 out/out_A64_64_R64_64_PE4_4"
#     ["config9"]="64 64 64 64 2 2 9 4 out/out_A64_64_R64_64_PE2_2"

#     ["config10"]="16 16 16 16 8 8 15 10 out/out_A16_16_R16_16_PE8_8"
#     ["config11"]="16 16 16 16 4 4 11 6 out/out_A16_16_R16_16_PE4_4"
#     ["config12"]="16 16 16 16 2 2 9 4 out/out_A16_16_R16_16_PE2_2"
    
# )
declare -A configs=(

    # ["config4"]="128 128 128 128 4 4 11 6 out/out_A128_128_R128_128_PE4_4"
    # ["config3"]="128 128 128 128 64 64 71 66 out/out_A128_128_R128_128_PE64_64" # crashed
    # ["config5"]="64 64 64 64 32 32 39 34 out/out_A64_64_R64_64_PE32_32"
    # ["config6"]="64 64 64 64 16 16 23 18 out/out_A64_64_R64_64_PE16_16"
    # ["config7"]="64 64 64 64 8 8 15 10 out/out_A64_64_R64_64_PE8_8"
    # ["config8"]="64 64 64 64 4 4 11 6 out/out_A64_64_R64_64_PE4_4"
    # ["config9"]="64 64 64 64 2 2 9 4 out/out_A64_64_R64_64_PE2_2"

    # ["config10"]="16 16 16 16 8 8 15 10 out/out_A16_16_R16_16_PE8_8"
    # ["config11"]="16 16 16 16 4 4 11 6 out/out_A16_16_R16_16_PE4_4"
    # ["config12"]="16 16 16 16 2 2 9 4 out/out_A16_16_R16_16_PE2_2"
    ["config13"]="4 4 4 4 4 4 11 6 out/out_A4_4_R4_4_PE4_4"
)


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

    # Run CSLC command
    cslc --arch=wse2 ./single_layer_layout.csl \
        --fabric-dims=${fabric_width},${fabric_height} \
        --fabric-offsets=${offset_x},${offset_y} \
        --params=layer_rows_A:${A_rows},layer_cols_A:${A_cols} \
        --params=layer_rows_R:${R_rows},layer_cols_R:${R_cols} \
        --params=layer_start_x:0,layer_start_y:0,layer_kernel_rows:${kernel_rows},layer_kernel_cols:${kernel_cols},layer_index:0 \
        --memcpy --channels=1 --max-inlined-iterations=1000000 -o ${out_name}

    # Run Python script
    cs_python single_layer_run.py --name ${out_name}
    
    echo "Completed configuration: $out_name"
    echo "Moving sim* files to out directory"
    mv sim* ${out_name}
    echo "----------------------------------------"
}

# Run all configurations
for config in "${!configs[@]}"; do
    read -r A_rows A_cols R_rows R_cols kernel_rows kernel_cols fabric_width fabric_height out_name <<< "${configs[$config]}"
    run_config "$A_rows" "$A_cols" "$R_rows" "$R_cols" "$kernel_rows" "$kernel_cols" "$fabric_width" "$fabric_height" "$out_name"
done

# Optional: Run a specific configuration by uncommenting and modifying the line below
# run_config 128 128 128 128 4 4 11 6 out_A128_128_R128_128_PE4_4
