#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
watch -n 1 python $SCRIPT_DIR/show_gpus.py "$@"