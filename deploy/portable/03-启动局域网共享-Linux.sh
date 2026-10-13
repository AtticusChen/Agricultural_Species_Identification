#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"
if [[ ! -x ./runtime/.runtime/bin/python ]]; then
  echo "检测到尚未安装运行环境，现在自动开始安装。"
  ./01-一键安装-Linux.sh
fi
export MIKANIA_HOST=0.0.0.0
exec ./runtime/.runtime/bin/python ./runtime/launcher.py
