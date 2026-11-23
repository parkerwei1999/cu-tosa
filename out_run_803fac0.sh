#!/bin/bash
cd /home/Parker/cu-tosa
PYTHONPATH=/home/Parker/cu-tosa CUDA_VISIBLE_DEVICES=0 /home/Parker/venv_et131_gpu/bin/python3 tests/graph_equiv.py --reps 5 $(cat /home/shared/Parker/u85_runs/cutosa_dtype_1123/graphs.txt) > out_graph_equiv_cas4k0.txt 2>&1
echo "rc=$?" >> out_graph_equiv_cas4k0.txt
PYTHON=/home/Parker/venv_et131_gpu/bin/python3 tests/run_ut_vs_reference.sh /home/shared/Parker/u85_runs/tosa_emu_ut_1123/tosa_emu_op_check.py $(grep -vE '^swin' /home/shared/Parker/u85_runs/cutosa_dtype_1123/graphs.txt | cut -d= -f2) > out_ut_vs_reference.txt 2>&1
