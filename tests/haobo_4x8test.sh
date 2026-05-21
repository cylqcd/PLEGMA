#!/usr/bin/env bash
# Cross-check test for meson_flowed_haobo on a 4^3 x 8 lattice.
# Corresponds to the original haobo_4x8test.sh used on the home machine.
#
# Usage:
#   EXEC=/path/to/meson_flowed_haobo CNF=/path/to/conf.0000 bash tests/haobo_4x8test.sh
#
# Both EXEC and CNF can also be set in the USER CONFIGURATION section below.
set -euo pipefail

# ============================= USER CONFIGURATION ===========================
# Path to the compiled meson_flowed_haobo executable:
: "${EXEC:?'EXEC is not set. Please set EXEC=/path/to/meson_flowed_haobo'}"

# Gauge configuration file:
: "${CNF:?'CNF is not set. Please set CNF=/path/to/conf.0000'}"
# ============================= END CONFIGURATION ============================

export QUDA_RESOURCE_PATH=.cache
export CUDA_DEVICE_MAX_CONNECTIONS=1
export QUDA_ENABLE_TUNING=0
export QUDA_TUNE_VERSION_CHECK=0
export QUDA_REORDER_LOCATION=GPU
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_GDR=0
ulimit -c 0

export OMP_NUM_THREADS=24

echo "CNF = ${CNF}"

L=4
T=8

PREC=double
PREC_SLOPPY=single
PREC_PRECON=single
PREC_NULL=single

RECON=18
RECON_SLOPPY=18
RECON_PRECON=18

echo "=== $(date) ==="

export ROTATEQ=1
export PIPLUSQ=1

set -x
KAONQ=true PIONQ=true mpirun -n 1 "$EXEC" \
   --procs 1 1 1 1 \
   --dims ${L} ${L} ${L} ${T} \
   --verbosity 2 \
   --load-gauge "${CNF}" \
   --nsrc 1 \
   --maxQsq 0 \
   --mu-s 0.2 \
   --corr-file-format hdf5 \
   --Q-dslash-type twisted-clover \
   --Q-prec ${PREC} --Q-prec-sloppy ${PREC_SLOPPY} \
   --Q-prec-precondition ${PREC_PRECON} --Q-prec-null ${PREC_NULL} \
   --Q-recon ${RECON} --Q-recon-sloppy ${RECON_SLOPPY} \
   --Q-recon-precondition ${RECON_PRECON} \
   --Q-inv-type cgne --Q-niter 5000 --Q-tol 1e-15 \
   --Q-use-mg false \
   --Q-kappa 0.11111111111111111 --Q-mu 0.112994350282 --Q-csw 1.74 \
   --tSinks 5 \
   --nsmear-APE 0 \
   --alpha-APE 0.5 \
   --nsmear-gauss 0 \
   --threep-filename threep.h5 \
   --twop-filename twop.h5 \
   --src-filename ./input.src

echo "=== done: $? ==="
