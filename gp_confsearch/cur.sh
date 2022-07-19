#!/usr/bin/bash

#!/bin/sh
#SBATCH --nodes=1
#SBATCH --tasks-per-node=24
#SBATCH --ntasks-per-node=24
#SBATCH --job-name=cs_orca
export KMP_STACKSIZE=10G
export OMP_STACKSIZE=10G
export OMP_NUM_THREADS=24,1
export MKL_NUM_THREADS=24

/opt/orca5/orca tests/cur.inp > tests/cur.out
