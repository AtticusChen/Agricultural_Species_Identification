#!/usr/bin/env bash
set -eu

cd -- "$(dirname -- "$0")"

platform="${1:-linux}"
root_dir="$(pwd)"
tools_dir="${root_dir}/.runtime-tools"
python_dir="${root_dir}/.runtime-python"
runtime_dir="${root_dir}/.runtime"
logs_dir="${root_dir}/logs"
log_file="${logs_dir}/install-${platform}.log"
uv_version="0.11.32"
uv_bin="${tools_dir}/uv"

mkdir -p "${tools_dir}" "${python_dir}" "${logs_dir}"
exec > >(tee -a "${log_file}") 2>&1

echo "============================================================"
echo "  薇甘菊识别系统 - 一键安装环境"
echo "============================================================"
echo "无需提前安装 Python，也不需要管理员权限。"
echo "首次安装需要联网，通常需要 5 至 15 分钟。"
echo

if [ ! -x "${uv_bin}" ]; then
  echo "[1/4] 准备环境安装工具……"
  installer_url="https://astral.sh/uv/${uv_version}/install.sh"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "${installer_url}" | env UV_UNMANAGED_INSTALL="${tools_dir}" UV_NO_MODIFY_PATH=1 sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "${installer_url}" | env UV_UNMANAGED_INSTALL="${tools_dir}" UV_NO_MODIFY_PATH=1 sh
  else
    echo "安装失败：电脑缺少下载工具 curl 或 wget。"
    echo "安装日志：${log_file}"
    exit 1
  fi
fi

if [ ! -x "${uv_bin}" ]; then
  echo "安装失败：环境安装工具没有正确下载。"
  echo "安装日志：${log_file}"
  exit 1
fi

export UV_PYTHON_INSTALL_DIR="${python_dir}"
export UV_LINK_MODE=copy
export UV_NO_MODIFY_PATH=1

echo "[2/4] 自动下载独立 Python 运行时……"
"${uv_bin}" python install 3.11

echo "[3/4] 创建系统专用运行环境……"
"${uv_bin}" venv "${runtime_dir}" --python 3.11 --managed-python --clear

echo "[4/4] 安装识别组件并检查模型……"
"${uv_bin}" pip install --python "${runtime_dir}/bin/python" -r "${root_dir}/requirements.txt"
"${runtime_dir}/bin/python" "${root_dir}/launcher.py" --self-test

{
  echo "Mikania portable runtime installed"
  echo "version=2.1.0"
  echo "python=3.11"
  date '+installed_at=%Y-%m-%dT%H:%M:%S%z'
} > "${root_dir}/.installed-ok"

echo
echo "[安装成功] 以后只需双击“02-启动识别系统”文件。"
