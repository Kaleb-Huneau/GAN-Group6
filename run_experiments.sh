#!/bin/bash

# claim it is not better to have them not the same
alpha_ds=(0.5 0.99999) #3
alpha_gs=(0.99999 1.1 10) # 0.25 0.50 32 64

# Required free memory in MiB
REQUIRED_FREE_MEMORY=2000  # Adjust this to your needs

# GPU index to check (1 for the second GPU)
GPU_INDEX=0

# Function to check available GPU memory on a specific GPU
check_gpu_memory() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_INDEX | awk '{print $1}'
}

while true; do
    AVAILABLE_MEMORY=$(check_gpu_memory)  # Capture the output of check_gpu_memory function

    echo "Available GPU memory on GPU $GPU_INDEX: $AVAILABLE_MEMORY MiB"

    if [ "$AVAILABLE_MEMORY" -ge "$REQUIRED_FREE_MEMORY" ]; then
        echo "Enough GPU memory available on GPU $GPU_INDEX. Starting training."
        # Loop through learning rates and batch sizes
        for alpha_d in "${alpha_ds[@]}"
        do
                for alpha_g in "${alpha_gs[@]}"
                do
                        echo "Running experiment with alpha_d=${alpha_d} and alpha_g=${alpha_g}"
                        python MRIALPHA128DataLoader.py --alpha_d ${alpha_d} --alpha_g ${alpha_g} --type 'notumor'
                done
        done
        break
        for alpha_d in "${alpha_ds[@]}"
        do
                for alpha_g in "${alpha_gs[@]}"
                do
                        echo "Running experiment with alpha_d=${alpha_d} and alpha_g=${alpha_g}"
                        python MRIALPHA128DataLoader.py --alpha_d ${alpha_d} --alpha_g ${alpha_g} --type 'onlytumors'
                done
        done
        break
    else
        echo "Not enough GPU memory available on GPU $GPU_INDEX. Waiting..."
        sleep 15  # Check again in 15 seconds
    fi
done