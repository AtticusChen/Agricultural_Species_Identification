#!/bin/zsh
set -euo pipefail

cd -- "$(dirname -- "$0")"
exec ./install_unix.sh macos
