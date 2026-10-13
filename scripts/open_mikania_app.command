#!/bin/zsh
set -euo pipefail

readonly LOCAL_PORT="18000"
readonly REMOTE_HOST="cx@10.33.231.250"
readonly APP_URL="http://127.0.0.1:${LOCAL_PORT}"

function keep_error_visible() {
  local mikania_exit_code=$?
  if (( mikania_exit_code != 0 )); then
    print -u2 -- "启动未完成。请保留这个窗口中的提示，不要直接打开 index.html。"
    if [[ -t 0 ]]; then
      read -r "?按回车键关闭窗口。"
    fi
  fi
}
trap keep_error_visible EXIT

function model_ready() {
  local mikania_response
  mikania_response=$(/usr/bin/curl --noproxy '*' --connect-timeout 3 --max-time 6 \
    --fail --silent "$@") || return 1
  [[ "$mikania_response" == *'"model"'* && "$mikania_response" == *mikania* ]] || return 1
  /usr/bin/grep -Eq '"status"[[:space:]]*:[[:space:]]*"ready"' <<< "$mikania_response"
}

if ! /usr/sbin/lsof -nP -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  typeset -a mikania_connection_options=()
  if ! model_ready "http://10.33.231.250:8000/api/v1/health"; then
    # A proxy TUN can intercept private addresses even when system-proxy exceptions exist.
    # Probe only this server via the default interface; never change system routes or VPN settings.
    mikania_interface=$(/sbin/route -n get default 2>/dev/null | /usr/bin/awk '/interface:/{print $2; exit}')
    if [[ -n "$mikania_interface" ]] && \
      model_ready --interface "$mikania_interface" "http://10.33.231.250:8000/api/v1/health"; then
      mikania_connection_options=(-B "$mikania_interface")
      print -- "已找到可用的内网连接路径，正在连接识别服务器。"
    else
      print -u2 -- "无法访问识别服务器。请确认已连接服务器所在内网或校园 VPN。"
      exit 1
    fi
  fi
  print -- "如出现 password 提示，请输入 cx 的服务器登录密码并按回车（输入时不显示字符）。"
  # This entry uses the project's server password, not an unrelated local key passphrase.
  # Keep these settings scoped to this connection; never alter the user's SSH config.
  /usr/bin/ssh \
    "${mikania_connection_options[@]}" \
    -fNT \
    -o PreferredAuthentications=password \
    -o PubkeyAuthentication=no \
    -o ExitOnForwardFailure=yes \
    -o ConnectTimeout=8 \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:8000" \
    "${REMOTE_HOST}"
fi

if ! model_ready "${APP_URL}/api/v1/health"; then
  print -u2 -- "本机入口未连接到就绪的薇甘菊模型，请检查服务器或网络；现有端口不会被强行关闭。"
  exit 1
fi

print -- "识别系统已就绪：${APP_URL}"
if [[ "${1:-}" != "--check" ]]; then
  /usr/bin/open "${APP_URL}"
fi
