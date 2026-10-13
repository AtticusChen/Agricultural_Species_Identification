#!/bin/zsh
set -u

cd -- "$(dirname -- "$0")"
chmod +x ./runtime/install_macos.command ./runtime/install_unix.sh
./runtime/install_macos.command
status=$?
echo
if [[ ${status} -eq 0 ]]; then
  echo "安装已经完成。请双击“02-启动识别系统-macOS.command”。"
else
  echo "安装没有完成，请检查网络后重试。安装日志位于 runtime/logs。"
fi
read -r "?按回车键关闭窗口。"
exit ${status}
