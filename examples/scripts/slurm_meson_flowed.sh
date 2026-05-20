#!/usr/bin/env bash
# =============================================================================
# Template: SLURM batch script for meson 2pt+3pt with Wilson flow (stochastic)
#
# Executable: meson_2pt_3pt_flowed_stoch  (built with PLEGMA_HIGHER_DERIV=ON)
#
# Usage:
#   1. Copy this file and adjust the variables in the USER CONFIGURATION section.
#   2. Submit a batch of gauge configs:
#        sbatch slurm_meson_flowed.sh 100 116 132 148
#      Each argument is a config number; 4 configs run in parallel on 4 GPUs.
#   3. The script launches one srun per GPU using CUDA_VISIBLE_DEVICES.
#
# The parallel loop at the bottom batches configs in groups of 4
# (one per GPU). Adjust gpu_id range if your node has fewer/more GPUs.
# =============================================================================

#SBATCH --job-name=meson_flowed
#SBATCH --partition=YOUR_PARTITION          # <-- set your partition
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --gpus=4
#SBATCH --time=5:30:00
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err
# #SBATCH --mail-user=YOUR@EMAIL            # uncomment if desired
# -----------------------------------------------------------------------
set -euo pipefail

# ============================= USER CONFIGURATION ===========================
# Paths — adjust all of these before running
export EXEC=~/PLEGMA_build/meson_2pt_3pt_flowed_stoch   # compiled executable
QUDA_SRC=~/src/quda                                      # for QUDA tuning cache key

export WORK_DIR=~/output/meson_flowed    # per-config output subdirs created here
export GAUGE_DIR=~/gauge_configs         # gauge field directory (conf.XXXX files)
export SRC_DIR=~/stoch_sources           # stochastic source files (src_XXXX.txt)

# Lattice dimensions
export L=32
export T=64

# Twisted-clover fermion parameters — adapt to your ensemble
export KAPPA=0.1400645
export CSW=1.74
export MUL=0.003        # light twisted mass
export MUS=0.022        # strange twisted mass

# Inversion tolerance
export INV_TOL=1.0e-7

# Momenta and output format
export QSQ_MAX=0
export DSLASH_TYPE=twisted-clover
export SOLVE_TYPE=direct-pc
export MASS_NORM=mass
export PIPELINE=24
export NGCRKRYLOV=24
export NITER=3000
export VERIFY=false

# Precision and reconstruction
export PREC=double
export PREC_SLOPPY=single
export PREC_PRECON=half
export RECON=12
export RECON_SLOPPY=8
export RECON_PRECON=8

# Smearing
export alphaGauss=0.2
export nsmearGauss=50
export alphaAPE=0.5
export nsmearAPE=50

# Stochastic source parameters
export NSAMPLES=1
export SEED=1234

# Multigrid setup
export MG_NU_PRE="0 0 1 0"
export MG_NU_POST="0 5 0 4"
export MG_SETUP_TOL="5e-7"
export MG_SETUP_ITER_0="0 1"
export MG_OMEGA=0.85
export MG_SETUP_TYPE='null'
export MG_PRE_ORTH=false
export MG_POST_ORTH=true
export MG_VERBOSITY=silent
export MG_LEVELS=2
export MG_N_VEC="0 24"
export MG_BLK_SZE_0="0 4 4 4 4"
export MG_MU_FACTOR_0="1 1.0"
export MG_COARSE_SOLVER_1='1 gcr'
export MG_COARSE_TOL_1='1 0.22'
export MG_COARSE_MAXITER_1='1 10'
export MG_SMOOTHER_TOL='0 0.25 1 0.25'
# ============================= END CONFIGURATION ============================

# QUDA tuning cache — keyed by QUDA git commit so different builds don't share
module load git 2>/dev/null || true
_quda_commit=$(git -C "$QUDA_SRC" rev-parse --short HEAD 2>/dev/null || echo "unknown")
export QUDA_RESOURCE_PATH="${WORK_DIR}/.cache_${_quda_commit}"
mkdir -p "$QUDA_RESOURCE_PATH"

# QUDA/GPU environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export QUDA_ENABLE_TUNING=1
export QUDA_TUNE_VERSION_CHECK=0
export QUDA_REORDER_LOCATION=GPU
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_GDR=0
ulimit -c 0

export OMP_NUM_THREADS=12

echo " "
echo "$(date)"

# ---------------------------------------------------------------------------
# run_one NUM4  — process a single gauge config on the GPU bound to
#                 CUDA_VISIBLE_DEVICES (set by the parallel loop below)
# ---------------------------------------------------------------------------
run_one() {
  local num="$1"
  local num4
  num4=$(printf "%04d" $((10#$num)))

  local workdir="${WORK_DIR}/${num4}"
  mkdir -p "$workdir"
  cd "$workdir"

  local logfile="meson_3pt_${num4}_$(date +%Y%m%d_%H%M%S).log"

  set -x
  srun --overlap -n 1 --gres=gpu:1 --cpus-per-task=12 \
    env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$EXEC" \
    --procs 1 1 1 1 \
    --dims ${L} ${L} ${L} ${T} \
    --verbosity 2 \
    --load-gauge "${GAUGE_DIR}/conf.${num4}" \
    --nsrc 1 \
    --maxQsq ${QSQ_MAX} --corr-file-format hdf5 \
    --src-filename "${SRC_DIR}/src_${num4}.txt" \
    --pion 1 \
    --kaon-uins 1 \
    --kaon-sins 1 \
    --t0-ref 2.39 \
    --nsmear-APE ${nsmearAPE} \
    --nsmear-gauss ${nsmearGauss} \
    --alpha-APE ${alphaAPE} \
    --alpha-gauss ${alphaGauss} \
    --Q-dslash-type ${DSLASH_TYPE} \
    --Q-prec ${PREC} \
    --Q-prec-sloppy ${PREC_SLOPPY} \
    --Q-recon ${RECON} \
    --Q-recon-sloppy ${RECON_SLOPPY} \
    --Q-recon-precondition ${RECON_PRECON} \
    --Q-kappa ${KAPPA} \
    --Q-mu ${MUL} \
    --Q-csw ${CSW} \
    --Q-mu-s ${MUS} \
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
    --tSinks 24 28 32 \
    --twop-filename twop_stoch \
    --threep-filename threep_stoch \
    --nsamples ${NSAMPLES} \
    --seed ${SEED} \
    2>&1 | tee "$logfile"
}

# ---------------------------------------------------------------------------
# Process configs in batches of 4, one per GPU
# ---------------------------------------------------------------------------
nums=( "$@" )
total=${#nums[@]}
idx=0

while [ $idx -lt $total ]; do
  pids=()
  for gpu_id in 0 1 2 3; do
    [ $idx -ge $total ] && break
    num=${nums[$idx]}
    CUDA_VISIBLE_DEVICES=$gpu_id run_one "$num" &
    pids+=("$!")
    idx=$(( idx + 1 ))
  done
  wait "${pids[@]}"
done
