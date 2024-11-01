#!/bin/bash

# Output directory
OUTPUT_DIR=/home/bricklib_dataflow/csl/csl-examples/tutorials/outputs

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Loop through each directory in the current directory
for dir in ./*; do
    # Check if commands.sh exists in the directory
    if [[ -f "$dir/commands.sh" ]]; then
        cd $dir
        echo "Running commands in $dir..."
        # Execute the commands.sh and capture output
        bash commands.sh 2>&1 | tee "$OUTPUT_DIR/${dir}_output.txt"
        echo "Output saved to $OUTPUT_DIR/${dir%/}_output.txt"
        cd ..
    else
        echo "No commands.sh found in $dir"
    fi
done

