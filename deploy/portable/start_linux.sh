#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"

if [[ ! -x .runtime/bin/python ]]; then
  echo "尚未安装运行环境，先执行自动安装。"
  ./install_linux.sh
fi

.runtime/bin/python launcher.py
