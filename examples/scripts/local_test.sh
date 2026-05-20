#!/usr/bin/env bash
# =============================================================================
# Template: local single-GPU test for qLoops_patterns
#
# Executable: qLoops_patterns  (built with PLEGMA_HIGHER_DERIV=ON)
#
# Usage:
#   ./local_test.sh [CONFIG_NUMBER]
#   e.g.  ./local_test.sh 100
#
# Requires prterun (or srun inside a SLURM allocation) on the PATH.
# Set the variables in the USER CONFIGURATION section before running.
# =============================================================================
set -euo pipefail

# ============================= USER CONFIGURATION ===========================
EXEC=~/PLEGMA_build/qLoops_patterns        # compiled executable
QUDA_SRC=~/src/quda                        # for QUDA tuning cache key

GAUGE_DIR=~/gauge_configs                  # gauge field directory (conf.XXXX files)
OUTDIR=~/output/qloops_patterns            # output goes here; per-config subdir created

# Lattice dimensions
L=24
T=48

# Twisted-clover parameters — adapt to your ensemble
KAPPA=0.1400645
CSW=1.74
MUL=0.0053

INV_TOL=1.0e-7
numSourcePositions=2

# Solver options
QSQ_MAX=0
DSLASH_TYPE=twisted-clover
MASS_NORM=mass
PIPELINE=24
NGCRKRYLOV=24
NITER=3000

PREC=double
PREC_SLOPPY=single
PREC_PRECON=half
RECON=12
RECON_SLOPPY=8
RECON_PRECON=8

# Multigrid setup
MG_NU_PRE="0 0 1 0"
MG_NU_POST="0 5 0 4"
MG_SETUP_TOL="5e-7"
MG_SETUP_ITER_0="0 1"
MG_OMEGA=0.85
MG_PRE_ORTH=false
MG_POST_ORTH=true
MG_VERBOSITY=silent
MG_LEVELS=2
MG_N_VEC="0 24"
MG_BLK_SZE_0="0 4 4 4 4"
MG_MU_FACTOR_0="1 1.0"
MG_COARSE_SOLVER_1='1 gcr'
MG_COARSE_TOL_1='1 0.22'
MG_COARSE_MAXITER_1='1 10'
MG_SMOOTHER_TOL='0 0.25 1 0.25'
# ============================= END CONFIGURATION ============================

# QUDA tuning cache — keyed by QUDA git commit so different builds don't share
_quda_commit=$(git -C "$QUDA_SRC" rev-parse --short HEAD 2>/dev/null || echo "unknown")
export QUDA_RESOURCE_PATH="${OUTDIR}/.cache_${_quda_commit}"
mkdir -p "$QUDA_RESOURCE_PATH"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export QUDA_ENABLE_TUNING=1
export QUDA_TUNE_VERSION_CHECK=0
export QUDA_REORDER_LOCATION=GPU
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_GDR=0
ulimit -c 0
export OMP_NUM_THREADS=4

# MPI launcher: prterun outside SLURM, srun inside
if [ -n "${SLURM_JOB_ID:-}" ]; then
  mpirun() { srun "$@"; }
else
  mpirun() { prterun "$@"; }
fi

# Config number from argument (default: 0)
N=${1:-0}
num4=$(printf "%04d" $((10#$N)))
CNF="${GAUGE_DIR}/conf.${num4}"

workdir="${OUTDIR}/${num4}"
mkdir -p "$workdir"
cd "$workdir"

echo "DIMS: L=${L} T=${T}"
echo "$(date)"

set -x
mpirun -n 1 "$EXEC" \
  --procs 1 1 1 1 \
  --dims ${L} ${L} ${L} ${T} \
  --load-gauge "${CNF}" \
  --verbosity 2 \
  --nsrc ${numSourcePositions} \
  --rng-seed 0 \
  --maxQsq ${QSQ_MAX} \
  --k-probing 1 \
  --spin-color-dil true \
  --corr-file-format hdf5 \
  --oneD-loops true \
  --twoD-loops true \
  --Q-dslash-type ${DSLASH_TYPE} \
  --Q-prec ${PREC} \
  --Q-prec-sloppy ${PREC_SLOPPY} \
  --Q-recon ${RECON} \
  --Q-recon-sloppy ${RECON_SLOPPY} \
  --Q-recon-precondition ${RECON_PRECON} \
  --Q-prec-precondition ${PREC_PRECON} \
  --Q-kappa ${KAPPA} \
  --Q-mu ${MUL} \
  --Q-csw ${CSW} \
  --Q-mass-normalization ${MASS_NORM} \
  --Q-pipeline ${PIPELINE} \
  --Q-ngcrkrylov ${NGCRKRYLOV} \
  --Q-niter ${NITER} \
  --Q-tol ${INV_TOL} \
  --Q-mg-levels ${MG_LEVELS} \
  --Q-mg-block-size ${MG_BLK_SZE_0} \
  --Q-mg-nu-pre ${MG_NU_PRE} \
  --Q-mg-nu-post ${MG_NU_POST} \
  --Q-mg-setup-tol ${MG_SETUP_TOL} \
  --Q-mg-mu-factor ${MG_MU_FACTOR_0} \
  --Q-mg-omega ${MG_OMEGA} \
  --Q-mg-setup-iters ${MG_SETUP_ITER_0} \
  --Q-mg-pre-orth ${MG_PRE_ORTH} \
  --Q-mg-post-orth ${MG_POST_ORTH} \
  --Q-mg-setup-inv 0 cg \
  --Q-mg-verbosity 0 ${MG_VERBOSITY} 1 ${MG_VERBOSITY} \
  --Q-mg-coarse-solver ${MG_COARSE_SOLVER_1} \
  --Q-mg-coarse-solver-tol ${MG_COARSE_TOL_1} \
  --Q-mg-coarse-solver-maxiter ${MG_COARSE_MAXITER_1} \
  --Q-mg-smoother-tol ${MG_SMOOTHER_TOL} \
  --Q-mg-eig-nKr 1 384 \
  --Q-mg-eig-nEv 1 256 \
  --Q-mg-eig-nConv 1 256 \
  --Q-mg-eig 0 false 1 true \
  --Q-mg-preserve-deflation true \
  2>&1 | tee "qloops_patterns_${num4}_$(date +%Y%m%d_%H%M%S).log"
