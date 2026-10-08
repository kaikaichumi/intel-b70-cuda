#!/bin/bash
# Container entry point for b70-ai: use the nearest project venv (.b70venv in
# the working directory or any parent), then run the command.
d=$PWD
while [ -n "$d" ] && [ "$d" != "/" ]; do
    if [ -x "$d/.b70venv/bin/python" ]; then
        export VIRTUAL_ENV="$d/.b70venv"
        export PATH="$VIRTUAL_ENV/bin:$PATH"
        break
    fi
    d=$(dirname "$d")
done
mkdir -p "${HOME:-/tmp}" 2>/dev/null
exec "$@"
