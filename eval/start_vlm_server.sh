#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1               
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00           
#SBATCH --output=vllm_server_%j.out

# Get the hostname allocated to this job by Slurm
NODE_NAME=$(hostname)
HOST_FILE=""


module purge
module load gcc/12.3.0
module load cuda/12.3.0
module load anaconda3

source activate vllm_eval_env 

# Write node name to the shared file
echo "$NODE_NAME" > "$HOST_FILE"
echo "vLLM server starting on node: $NODE_NAME"

export XDG_CACHE_HOME=""
export TRITON_CACHE_DIR=""
export HF_HOME=""

# Clean up environment loading (Prefix with Spack/Apptainer if required by your HPC)
# Example: apptainer exec --nv your_container.sif ...
python -m vllm.entrypoints.openai.api_server \
	--model Qwen/Qwen2-VL-72B-Instruct \ 
	--tensor-parallel-size 1 \
	--port 8000 \
	--trust-remote-code \
	--max-model-len 8192
