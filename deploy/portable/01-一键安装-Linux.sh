#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"
chmod +x ./runtime/install_linux.sh ./runtime/install_unix.sh
./runtime/install_linux.sh
