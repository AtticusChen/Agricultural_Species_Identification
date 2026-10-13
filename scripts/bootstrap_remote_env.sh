#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-/home/cx/Agricultural_Species_Identification}"
env_prefix="${2:-/home/cx/.conda/envs/mikania-detect}"

if [[ ! -d "${project_root}" ]]; then
  echo "Project directory not found: ${project_root}" >&2
  exit 1
fi

if [[ ! -x "${env_prefix}/bin/python" ]]; then
  conda create --yes --prefix "${env_prefix}" python=3.11 pip
fi

conda run --prefix "${env_prefix}" python -m pip install --upgrade pip
conda run --prefix "${env_prefix}" python -m pip install \
  torch==2.13.0 torchvision==0.28.0
conda run --prefix "${env_prefix}" python -m pip install \
  --editable "${project_root}[train,dev]"

mkdir -p "${project_root}/environment"
conda run --prefix "${env_prefix}" python "${project_root}/scripts/verify_training_env.py" \
  > "${project_root}/environment/remote-training-env.json"
conda run --prefix "${env_prefix}" python -m pip freeze \
  > "${project_root}/environment/remote-pip-freeze.txt"

echo "Environment ready: ${env_prefix}"
echo "Snapshot: ${project_root}/environment/remote-training-env.json"
