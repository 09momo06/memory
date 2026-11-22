#!/usr/bin/env bash
# Source before experiment commands; all packages, caches and temporary files stay here.
export DMEM_ROOT=/home/lj/wtr/dmem
export PATH=/home/lj/miniconda3/bin:$PATH
export CONDA_ENVS_PATH="$DMEM_ROOT/.conda/envs"
export CONDA_PKGS_DIRS="$DMEM_ROOT/.conda/pkgs"
export CONDA_REGISTER_ENVS=false
export CONDARC="$DMEM_ROOT/.condarc"
export XDG_CACHE_HOME="$DMEM_ROOT/.cache"
export PIP_CACHE_DIR="$DMEM_ROOT/.cache/pip"
export PIP_NO_CACHE_DIR=1
export PIP_TIMEOUT=90
export PIP_RETRIES=20
export PIP_RESUME_RETRIES=50
export HF_HOME="$DMEM_ROOT/.cache/huggingface"
export TORCH_HOME="$DMEM_ROOT/.cache/torch"
export TMPDIR="$DMEM_ROOT/.tmp"
export PYTHONPYCACHEPREFIX="$DMEM_ROOT/.cache/pycache"
export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$TMPDIR" "$XDG_CACHE_HOME" "$DMEM_ROOT/logs"
