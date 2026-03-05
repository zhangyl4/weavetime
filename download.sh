#!/bin/bash

# Function to run command 10 times regardless of errors
run_command_10_times() {
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt of $max_attempts: Downloading dataset..."
        
        huggingface-cli download --repo-type dataset --resume-download Becomebright/QAEgo4D-MC-test # VLM-Reasoning/VCR-Bench lmms-lab/LLaVA-Video-178K
        
        echo "Attempt $attempt completed."
        sleep 2
        ((attempt++))
    done
    
    echo "Completed all $max_attempts attempts."
}

# Start the download process
run_command_10_times
