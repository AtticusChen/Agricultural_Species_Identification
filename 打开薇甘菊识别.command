#!/bin/zsh
# Stable desktop entry: open the actual service, never the HTML source file.
readonly MIKANIA_PROJECT_DIR="${0:A:h}"
exec /bin/zsh "${MIKANIA_PROJECT_DIR}/scripts/open_mikania_app.command" "$@"
