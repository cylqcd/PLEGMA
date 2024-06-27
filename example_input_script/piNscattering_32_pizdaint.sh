#!/bin/bash -l
#SBATCH --mail-user=pittler@hiskp.uni-bonn.de
#SBATCH --job-name=CONFID_PINFACT
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH --contiguous
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --account=pr79
#SBATCH --cpus-per-task=12

N=CONFID
HOME_DIR=$(pwd)
VERSION=0
NUM_THREADS=12
GPUPERNODE=1
NNODES=64

module load daint-gpu
module unload PrgEnv-cray
module load PrgEnv-gnu
module load cudatoolkit
module load CMake
module load cray-hdf5-parallel
#source PizDaint_load_modules.sh

gdr=0
p2p=0
async=0
mempool=0

machine_id=PizDaint
quda_label=quda_1.0.x-dynamic_clover
quda_commit=e92ebd2e576691be6a5bea8a9b6e406b23f31246
gpu_arch=sm_60

export QUDA_RESOURCE_PATH=$(pwd)/${machine_id}-${quda_label}-${quda_commit}-${gpu_arch}_gdr${gdr}_p2p${p2p}
if [ ! -d ${QUDA_RESOURCE_PATH} ]; then
  mkdir -p ${QUDA_RESOURCE_PATH}
fi

export CRAY_CUDA_MPS=1
export OMP_NUM_THREADS=${NUM_THREADS}
ulimit -c 0
echo " "

export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_DSLASH_COARSE_POLICY=0

export GOMP_CPU_AFFINITY=0-23:2
export QUDA_RESOURCE_PATH=${QUDA_RESOURCE_PATH}
export OMP_NUM_THREADS=12
export QUDA_ENABLE_GDR=${gdr}
export QUDA_ENABLE_P2P=${p2p}
export QUDA_ENABLE_TUNING=1
export QUDA_ENABLE_DEVICE_MEMORY_POOL=${mempool}
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_NEMESIS_ASYNC_PROGRESS=${async}

DIAGPREFIX="Diagramm"
DATA_DIR="data"
OUTVECTOR="${DATA_DIR}/vector"

QUDA_BIN=/scratch/snx3000/fpittler/piN_factorized_I32/cB211a.072.64/piNdiagrams
SCRATCH_DIR=$(pwd)/
CORR_DIR=$(pwd)/
mkdir -p ${CORR_DIR}
CNF=/scratch/snx3000/mpetschl/cB211a.072.64/configs/conf.${N}

L=64
T=128

KAPPA=0.1394265
CSW=1.69
MUL=0.00072
INV_TOL=1.0e-9


QSQ_MAX=0
DSLASH_TYPE=twisted-clover
SOLVE_TYPE=direct-pc
MASS_NORM=mass
PIPELINE=24
NGCRKRYLOV=24
NITER=1000
VERIFY=false
TRAJ=$N

PREC=double
PREC_SLOPPY=single
PREC_PRECON=half
RECON=12
RECON_SLOPPY=8
RECON_PRECON=8

UseFullOp=false
UseEven=true

numSourcePositions=16
pathListSourcePositions=${HOME_DIR}/${N}.src
pathListMomenta=${HOME_DIR}/combination_pi2_pf1_pf2

# smearing parameters
alphaGauss=1.
nsmearAPE=60
nsmearGauss=140
alphaAPE=0.5
# File options
CORR_FILE_FORMAT=hdf5
CHECK_CORR_FILES=yes
CORR_WRITE_SPACE=momentum
# High mom format in yet implemented in the plugin
#HighMomForm=yes
VERBOSITY_LEVEL=verbose

# Determine number of nodes, gridsizes and dims
#
ZGRID=4
YGRID=2
TGRID=4
XGRID=2

XDIM=$((L/XGRID))
YDIM=$((L/YGRID))
ZDIM=$((L/ZGRID))
TDIM=$((T/TGRID))

#MG options
MG_NU_PRE="0 1 0"
MG_NU_POST="4 1 2"
MG_SETUP_TOL="5e-7"
MG_SETUP_ITER_0="0 1"
MG_SETUP_ITER_1="1 1"
MG_OMEGA=0.85
MG_SETUP_TYPE='null'
MG_PRE_ORTH=false
MG_POST_ORTH=true
MG_VERBOSITY=silent

MG_LEVELS=3
#Add more here for >2 levels
MG_N_VEC_0=" 24"
MG_BLK_SZE_0="0 4 4 4 8 1 2 2 2 2"
MG_MU_FACTOR_0="2 1.0"

MG_COARSE_SOLVER_1='2 ca-gcr'
MG_COARSE_TOL_1='1 0.22 2 0.22'
MG_COARSE_MAXITER_1='2 50'


echo "GRID(X,Y,Z,T) = ${XGRID} , ${YGRID} , ${ZGRID} , ${TGRID}"
echo "DIM(X,Y,Z,T)  = ${XDIM} , ${YDIM} , ${ZDIM} , ${TDIM}"
echo " "

echo `date`

RUN_COMMAND="srun -n $((NNODES*GPUPERNODE)) --ntasks-per-node=1 -c 12 ${QUDA_BIN} \
--procs ${XGRID} ${YGRID} ${ZGRID} ${TGRID} \
--dims ${XDIM} ${YDIM} ${ZDIM} ${TDIM} \
--verbosity 2  \
--load-gauge ${CNF} \
--nsrc ${numSourcePositions} \
--src-filename ${pathListSourcePositions} \
--nsmear-APE ${nsmearAPE} \
--nsmear-gauss ${nsmearGauss} \
--alpha-APE ${alphaAPE} \
--momlist-filename combination_pi2_pf1_pf2 \
--time-dilution true \
--nstochSamples 12 \
--alpha-gauss ${alphaGauss} \
--outdiagramPrefix ${DIAGPREFIX} \
--Q-dslash-type ${DSLASH_TYPE} \
--Q-prec ${PREC} \
--Q-prec-sloppy ${PREC_SLOPPY} \
--Q-recon ${RECON} \
--Q-recon-sloppy ${RECON_SLOPPY} \
--Q-recon-precondition ${RECON_PRECON} \
--Q-kappa ${KAPPA} \
--Q-mu ${MUL} \
--Q-csw ${CSW} \
--Q-mass-normalization ${MASS_NORM} \
--Q-pipeline ${PIPELINE} \
--Q-ngcrkrylov ${NGCRKRYLOV} \
--Q-niter ${NITER} \
--Q-use-mg true \
--Q-inv-type gcr \
--Q-tol 1e-09 \
--Q-mg-levels 3 \
--Q-mg-block-size 0 4 4 4 8 1 2 2 2 2 \
--Q-mg-nu-pre 0 0 1 0 \
--Q-mg-nu-post 0 4 1 2 \
--Q-mg-setup-tol 5e-7 \
--Q-mg-setup-maxiter 1000 \
--Q-mg-setup-inv 0 cg 1 cg \
--Q-mg-mu-factor 2 1. \
--Q-mg-omega 0.85 \
--Q-mg-setup-iters 0 1 \
--Q-mg-pre-orth false \
--Q-mg-post-orth true \
--Q-mg-verbosity 0 silent 1 silent 2 silent \
--Q-mg-smoother 0 ca-gcr 1 ca-gcr \
--Q-mg-smoother-tol 0 0.22 1 0.46 \
--Q-mg-nvec 0 24 1 32 \
--Q-mg-coarse-solver 2 ca-gcr \
--Q-mg-coarse-solver-tol 1 0.22 0.22 \
--Q-mg-coarse-solver-maxiter 2 50 \
--Q-mg-eig-nConv 2 800 \
--Q-mg-eig 2 true \
--Q-mg-eig-type 2 trlm \
--Q-mg-eig-nEv 2 800 \
--Q-mg-eig-nKr 2 1200 \
--Q-mg-eig-tol 2 1e-4 \
--Q-mg-eig-poly-deg 2 100 \
--Q-mg-eig-amin 2 4e-2 \
--Q-mg-eig-amax 2 8.0 \
--Q-mg-eig-max-restarts 2 25 \
--Q-mg-eig-use-dagger 2 false \
--Q-mg-eig-use-normop 2 true \
--Q-mg-preserve-deflation true \
"
echo "Run command is:"
echo ${RUN_COMMAND}
echo `date`
eval ${RUN_COMMAND}
echo `date`

pkill -u sbacchio
