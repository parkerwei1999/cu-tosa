#!/bin/bash
# The op-level test against Arm's tosa_reference_model (written for the kit's tosa_emu), run on the cu-tosa emulator
# through tests/shim. UT = path of tosa_emu_op_check.py; graphs = the .tosa files to take real operator instances from.
# usage: tests/run_ut_vs_reference.sh <tosa_emu_op_check.py> graph.tosa ...
here=$(cd "$(dirname "$0")/.." && pwd)
ut=$1; shift
export PYTHONPATH=$here/tests/shim:$here CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
${PYTHON:-python3} "$ut" --devices cuda --per-signature 3 --graphs "$@"
rc=$?
echo "rc=$rc"
