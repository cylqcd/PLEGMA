#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <quda_types.h>
#include <comm_quda.h>

#ifdef MULTI_GPU
int device = -1;
#else
int device = 0;
#endif

QudaReconstructType link_recon = QUDA_RECONSTRUCT_NO;
QudaReconstructType link_recon_sloppy = QUDA_RECONSTRUCT_INVALID;
QudaReconstructType link_recon_precondition = QUDA_RECONSTRUCT_INVALID;
QudaPrecision prec = QUDA_SINGLE_PRECISION;
QudaPrecision  prec_sloppy = QUDA_INVALID_PRECISION;
QudaPrecision  prec_precondition = QUDA_INVALID_PRECISION;
QudaPrecision prec_null = QUDA_INVALID_PRECISION;
int xdim = 24;
int ydim = 24;
int zdim = 24;
int tdim = 24;
int Lsdim = 16;
QudaDagType dagger = QUDA_DAG_NO;
int gridsize_from_cmdline[4] = {1,1,1,1};
QudaDslashType dslash_type = QUDA_WILSON_DSLASH;
char latfile[256] = "";
int Nsrc = 1;
int Msrc = 1;
int niter = 100;
int gcrNkrylov = 10;
int pipeline = 0;
int solution_accumulator_pipeline = 0;
int test_type = 0;
int nvec[QUDA_MAX_MG_LEVEL] = { };
char vec_infile[256] = "";
char vec_outfile[256] = "";
QudaInverterType inv_type;
QudaInverterType precon_type = QUDA_INVALID_INVERTER;
int multishift = 0;
bool verify_results = true;
double mass = 0.1;
double kappa = -1.0;
double mu = 0.1;
double anisotropy = 1.0;
double clover_coeff = 0.1;
bool compute_clover = false;
double tol = 1e-7;
double tol_hq = 0.1;
QudaTwistFlavorType twist_flavor = QUDA_TWIST_SINGLET;
bool kernel_pack_t = false;
QudaMassNormalization normalization = QUDA_KAPPA_NORMALIZATION;
QudaMatPCType matpc_type = QUDA_MATPC_EVEN_EVEN;
QudaSolveType solve_type = QUDA_DIRECT_PC_SOLVE;

int mg_levels = 2;

int nu_pre = 2;
int nu_post = 2;
double mu_factor[QUDA_MAX_MG_LEVEL] = { };
QudaVerbosity mg_verbosity[QUDA_MAX_MG_LEVEL] = { };
QudaInverterType setup_inv[QUDA_MAX_MG_LEVEL] = { };
int num_setup_iter[QUDA_MAX_MG_LEVEL] = { };//
double setup_tol = 5e-6;
QudaSetupType setup_type = QUDA_NULL_VECTOR_SETUP;//
bool pre_orthonormalize = false;//
bool post_orthonormalize = true;//
double omega = 0.85;
QudaInverterType coarse_solver[QUDA_MAX_MG_LEVEL] = { };
double coarse_solver_tol[QUDA_MAX_MG_LEVEL] = { };
QudaInverterType smoother_type[QUDA_MAX_MG_LEVEL] = { };
double smoother_tol[QUDA_MAX_MG_LEVEL] = { };
int coarse_solver_maxiter[QUDA_MAX_MG_LEVEL] = { };
bool generate_nullspace = true;
bool generate_all_levels = true;
QudaSchwarzType schwarz_type[QUDA_MAX_MG_LEVEL] = { };
int schwarz_cycle[QUDA_MAX_MG_LEVEL] = { };

int geo_block_size[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM] = { };

int dim_partitioned[4] = {0,0,0,0};

/////////////////////
// QKXTM additions //
/////////////////////

//-C.K. Generic Input parameters
int traj;
char latfile_smeared[257] = "";
double csw = 1.57551;
char verbosity_level[257] = "summarize";
bool isEven = true;

//-C.K. Correlation functions input parameters
int src[4] = {0,0,0,0};
int Ntsink = 1;
char pathList_tsink[257] = "list_tsinksource.txt";
int Q_sq = 0;
int nsmearAPE = 20;
int nsmearGauss = 50;
double alphaAPE = 0.5;
double alphaGauss = 4.0;

char corr_dirname[1024] = "./";
char twop_filename[257] = "twop";
char threep_filename[257] = "threep";
char prop_path[257] = "prop";

char pathListRun3pt[257] = "listrun3pt.txt";
char run3pt[257] = "all";
char check_file_exist[257] = "no";
char *corr_file_format = (char*)"ASCII";

int numSourcePositions = 1;
char pathListSourcePositions[257] = "listSourcePositions.txt";

int Nproj = 1;
char proj_list_file[257] = "default";

char *corr_write_space =(char*) "MOMENTUM";

//-C.K. loop Parameters
int Nstoch = 100;              // Number of stochastic noise vectors
int Ndump  = 10;               // Write the loop every Ndump stoch. vectors
unsigned long int seed = 100;  // The seed for the stochastic vectors
char loop_fname[512] = "loop";
char *loop_file_format =(char*) "ASCII";
char source_type[257] = "random";

#ifdef HAVE_ARPACK
//- Loop params with ARPACK enabled
int defl_steps = 1;
int defl_step_nEv[10] = { };

//-C.K. ARPACK Parameters
char pathEigenVectorsUp[257] = "ev_u.0000";
char pathEigenVectorsDown[257] = "ev_d.0000";
char pathEigenValuesUp[257] = "evals_u.dat";
char pathEigenValuesDown[257] = "evals_d.dat";

int PolyDeg = 100;         // degree of the Chebysev polynomial
int nEv = 100;             // Number of the eigenvectors we want
int nKv = 200;             // total size of Krylov space
char *spectrumPart = "SR"; // for which part of the spectrum we want to solve
bool isACC = true;
double tolArpack = 1.0e-5;
int maxIterArpack = 100000;
int modeArpack = 1;
char arpack_logfile[512] = "arpack.log";
double amin = 3.0e-4;
double amax = 3.5;
#endif
bool isFullOp = false;
int k_probing = 0; // default is without probing
bool spinColorDil = false;

// these variables will allow us to do the probing in different runs
// if you plan to do the hadamard vectors all in one run then this is not needed
int hadamLow = 0;
int hadamHigh = 0;
//===========//

void __attribute__((weak)) usage_extra(char** argv){};

// function called before reading the params
static void set_default_values(){
  // We give here the default value to some of the array
  for(int i =0; i<QUDA_MAX_MG_LEVEL; i++) {
    mg_verbosity[i] = QUDA_SILENT;
    setup_inv[i] = QUDA_BICGSTAB_INVERTER;
    num_setup_iter[i] = 1;
    mu_factor[i] = 1.;
    schwarz_type[i] = QUDA_INVALID_SCHWARZ;
    schwarz_cycle[i] = 1;
    smoother_type[i] = QUDA_MR_INVERTER;
    smoother_tol[i] = 0.25;
    coarse_solver[i] = QUDA_GCR_INVERTER;
    coarse_solver_tol[i] = 0.25;
    coarse_solver_maxiter[i] = 10;
  }
}

// function call as last after all params have been read
static void finalize_values(){
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  if (dslash_type != QUDA_TWISTED_MASS_DSLASH && 
      dslash_type != QUDA_TWISTED_CLOVER_DSLASH){
    printfQuda("This test is only for twisted mass or twisted clover operator\n");
    exit(-1);
  }

}

// TODO: Everything should be done in the meanwhile we read the command line
static void set_PLEGMA_params(plegma::PLEGMA_params *params){
  params->nsmearAPE = nsmearAPE;
  params->nsmearGauss = nsmearGauss;
  params->alphaAPE = alphaAPE;
  params->alphaGauss = alphaGauss;
  params->isEven = isEven;
  params->lL[0] = xdim;
  params->lL[1] = ydim;
  params->lL[2] = zdim;
  params->lL[3] = tdim;
  for(int i=0; i<4; i++)
    params->procs[i] = gridsize_from_cmdline[i];
  params->Nsources = numSourcePositions;
  params->Q_sq = Q_sq;
  params->traj = traj;

  if(strcmp(check_file_exist,"yes")==0 || 
     strcmp(check_file_exist,"YES")==0 )  params->check_files = true;
  else
    params->check_files = false;
  
  // Determine whether to write the correlation functions in ASCII or HDF5 
  // format
  if( strcmp(corr_file_format,"ASCII")==0 || 
      strcmp(corr_file_format,"ascii")==0 ) 
    params->CorrFileFormat = ASCII_FORM;   
  else if( strcmp(corr_file_format,"HDF5")==0 || 
	   strcmp(corr_file_format,"hdf5")==0 ) 
    params->CorrFileFormat = HDF5_FORM; 
  else
    fprintf(stderr,"Undefined option for --corr_file_format. Options are ASCII(ascii)/HDF5(hdf5)\n");
  params->corr_dir = corr_dirname;
  
  // Determine for which source-positions to run for the 3pt
  if(strcmp(run3pt,"all")==0 || 
     strcmp(run3pt,"ALL")==0) {
    for(int is = 0; is < numSourcePositions; is++) params->run3pt_src[is] = 1;
  }
  else if(strcmp(run3pt,"none")==0 || 
	  strcmp(run3pt,"NONE")==0) {
    for(int is = 0; is < numSourcePositions; is++) params->run3pt_src[is] = 0;
  }
  else if(strcmp(run3pt,"file")==0 || 
	  strcmp(run3pt,"FILE")==0) {
    printfQuda("Will read from file %s for which source-positions to perform the three-point function\n",pathListRun3pt);  
    FILE *ptr_run3pt;
    ptr_run3pt = fopen(pathListRun3pt,"r");
    if(ptr_run3pt == NULL) {
      fprintf(stderr,"Error opening file %s \n",pathListRun3pt);
      exit(-1);
    }
    for(int is = 0; is < numSourcePositions; is++) 
      fscanf(ptr_run3pt,"%d\n",&(params->run3pt_src[is]));
    fclose(ptr_run3pt);
  }
  else {
    for(int is = 0; is < numSourcePositions; is++) params->run3pt_src[is] = 1;
  }

  //-C.K: Get the list of source positions
  FILE *ptr_sources;
  ptr_sources = fopen(pathListSourcePositions,"r");
  if(ptr_sources == NULL){
    fprintf(stderr,"Error open file to read the source positions\n");
    exit(-1);
  }
  for(int is = 0 ; is < numSourcePositions ; is++)
    fscanf(ptr_sources,"%d %d %d %d",
	   &(params->sourcePosition[is][0]),
	   &(params->sourcePosition[is][1]), 
	   &(params->sourcePosition[is][2]), 
	   &(params->sourcePosition[is][3]));  
  fclose(ptr_sources);

    
  //-C.K: Read in the sink-source separations
  params->Ntsink = Ntsink;
  FILE *ptr_tsink;
  ptr_tsink = fopen(pathList_tsink,"r");
  if(ptr_tsink == NULL){
    fprintf(stderr,"Error opening file for sink-source separations\n");
    exit(-1);
  }
  for(int it = 0 ; it < Ntsink ; it++) {
    fscanf(ptr_tsink,"%d\n", &(params->tsinkSource[it]));
  }
  fclose(ptr_tsink);
  
  //-C.K: Determine for which projectors to run for the 3pt
  if(strcmp(proj_list_file,"default")==0){
    for(int i=0;i<Ntsink;i++){
      params->proj_list[i][0] = 0;   // Do only the G4 projector for all tsink's
      params->Nproj[i] = Nproj; // Nproj = 1 by default
    }
  }
  else{
    FILE *proj_ptr;
    char *proj_file;
    for(int it=0;it<Ntsink;it++){
      asprintf(&proj_file,"%s_tsink%d.txt",proj_list_file,
	       params->tsinkSource[it]);
      if( (proj_ptr = fopen(proj_file,"r")) == NULL ) {
	fprintf(stderr,"Cannot open projector file %s for reading.\n Hint: Make sure that 1: it ends as _tsink%d.txt and 2: the input passed is truncated up to this string.\n",proj_file,params->tsinkSource[it]);
	exit(-1);
      }
      
      fscanf(proj_ptr,"%d",&(params->Nproj[it]));
      for(int p=0;p<params->Nproj[it];p++) fscanf(proj_ptr,"%d\n",
					       &(params->proj_list[it][p]));
      
      fclose(proj_ptr);
    }
  }
  
  // Determine whether to write the correlation functions in position or 
  // momentum space
  if( strcmp(corr_write_space,"MOMENTUM")==0 || 
      strcmp(corr_write_space,"momentum")==0 ) 
    params->CorrSpace = MOMENTUM_SPACE;      
  else if( strcmp(corr_write_space,"POSITION")==0 || 
	   strcmp(corr_write_space,"position")==0 ) 
    params->CorrSpace = POSITION_SPACE; 
  else fprintf(stderr,"Undefined option for --corr_write_space. Options are MOMENTUM(momentum)/POSITION(position)\n");

}

static void usage(char** argv )
{  
  printf("Usage: %s [options]\n", argv[0]);
  printf("Common options: \n");
#ifndef MULTI_GPU
  printf("    --device <n>                              # Set the CUDA device to use (default 0, single GPU only)\n");     
#endif
  printf("    --prec <double/single/half>               # Precision in GPU\n");
  printf("    --prec-sloppy <double/single/half>        # Sloppy precision in GPU\n");
  printf("    --prec-precondition <double/single/half>  # Preconditioner precision in GPU\n");
  printf("    --prec-null <double/single/half>          # Null vector precision in GPU\n");
  printf("    --recon <8/9/12/13/18>                    # Link reconstruction type\n");
  printf("    --recon-sloppy <8/9/12/13/18>             # Sloppy link reconstruction type\n");
  printf("    --recon-precondition <8/9/12/13/18>       # Preconditioner link reconstruction type\n");
  printf("    --dagger                                  # Set the dagger to 1 (default 0)\n"); 
  printf("    --dim <n>                                 # Set space-time dimension (X Y Z T)\n"); 
  printf("    --sdim <n>                                # Set space dimension(X/Y/Z) size\n"); 
  printf("    --xdim <n>                                # Set X dimension size(default 24)\n");     
  printf("    --ydim <n>                                # Set X dimension size(default 24)\n");     
  printf("    --zdim <n>                                # Set X dimension size(default 24)\n");     
  printf("    --tdim <n>                                # Set T dimension size(default 24)\n");  
  printf("    --Lsdim <n>                               # Set Ls dimension size(default 16)\n");  
  printf("    --gridsize <x y z t>                      # Set the grid size in all four dimension (default 1 1 1 1)\n");
  printf("    --xgridsize <n>                           # Set grid size in X dimension (default 1)\n");
  printf("    --ygridsize <n>                           # Set grid size in Y dimension (default 1)\n");
  printf("    --zgridsize <n>                           # Set grid size in Z dimension (default 1)\n");
  printf("    --tgridsize <n>                           # Set grid size in T dimension (default 1)\n");
  printf("    --partition <mask>                        # Set the communication topology (X=1, Y=2, Z=4, T=8, and combinations of these)\n");
  printf("    --kernel-pack-t                           # Set T dimension kernel packing to be true (default false)\n");
  printf("    --dslash-type <type>                      # Set the dslash type, the following values are valid\n"
	 "                                                  wilson/clover/twisted-mass/twisted-clover/staggered\n"
         "                                                  /asqtad/domain-wall/domain-wall-4d/mobius\n");
  printf("    --flavor <type>                           # Set the twisted mass flavor type (singlet (default), deg-doublet, nondeg-doublet)\n");
  printf("    --load-gauge file                         # Load gauge field \"file\" for the test (requires QIO)\n");
  printf("    --niter <n>                               # The number of iterations to perform (default 10)\n");
  printf("    --ngcrkrylov <n>                          # The number of inner iterations to use for GCR, BiCGstab-l (default 10)\n");
  printf("    --pipeline <n>                            # The pipeline length for fused operations in GCR, BiCGstab-l (default 0, no pipelining)\n");
  printf("    --solution-pipeline <n>                   # The pipeline length for fused solution accumulation (default 0, no pipelining)\n");
  printf("    --inv-type <cg/bicgstab/gcr>              # The type of solver to use (default cg)\n");
  printf("    --precon-type <mr/ (unspecified)>         # The type of solver to use (default none (=unspecified)).\n");
  printf("    --multishift <true/false>                 # Whether to do a multi-shift solver test or not (default false)\n");
  printf("    --mass                                    # Mass of Dirac operator (default 0.1)\n");
  printf("    --kappa                                   # Kappa of Dirac operator (default -1.0)\n");
  printf("    --mu                                      # Twisted-Mass of Dirac operator (default 0.1)\n");
  printf("    --compute-clover                          # Compute the clover field or use random numbers (default false)\n");
  printf("    --clover-coeff                            # Clover coefficient (default 1.0)\n");
  printf("    --anisotropy                              # Temporal anisotropy factor (default 1.0)\n");
  printf("    --mass-normalization                      # Mass normalization (kappa (default) / mass / asym-mass)\n");
  printf("    --matpc                                   # Matrix preconditioning type (even-even, odd-odd, even-even-asym, odd-odd-asym) \n");
  printf("    --solve-type                              # The type of solve to do (direct, direct-pc, normop, normop-pc, normerr, normerr-pc) \n");
  printf("    --tol  <resid_tol>                        # Set L2 residual tolerance\n");
  printf("    --tolhq  <resid_hq_tol>                   # Set heavy-quark residual tolerance\n");
  printf("    --test                                    # Test method (different for each test)\n");
  printf("    --verify <true/false>                     # Verify the GPU results using CPU results (default true)\n");
  printf("    --mg-nvec <level nvec>                    # Number of null-space vectors to define the multigrid transfer operator on a given level\n");
  printf("    --mg-gpu-prolongate <true/false>          # Whether to do the multigrid transfer operators on the GPU (default false)\n");
  printf("    --mg-levels <2+>                          # The number of multigrid levels to do (default 2)\n");
  printf("    --mg-nu-pre  <1-20>                       # The number of pre-smoother applications to do at each multigrid level (default 2)\n");
  printf("    --mg-nu-post <1-20>                       # The number of post-smoother applications to do at each multigrid level (default 2)\n");
  printf("    --mg-setup-inv <level inv>                # The inverter to use for the setup of multigrid (default bicgstab)\n");
  printf("    --mg-setup-iters <level iter>             # The number of setup iterations to use for the multigrid (default 1)\n");
  printf("    --mg-setup-tol                            # The tolerance to use for the setup of multigrid (default 5e-6)\n");
  printf("    --mg-setup-type <null/test>               # The type of setup to use for the multigrid (default null)\n");
  printf("    --mg-pre-orth <true/false>                # If orthonormalize the vector before inverting in the setup of multigrid (default false)\n");
  printf("    --mg-post-orth <true/false>               # If orthonormalize the vector after inverting in the setup of multigrid (default true)\n");
  printf("    --mg-omega                                # The over/under relaxation factor for the smoother of multigrid (default 0.85)\n");
  printf("    --mg-coarse-solver <level gcr/etc.>       # The solver to wrap the V cycle on each level (default gcr, only for levels 1+)\n");
  printf("    --mg-coarse-solver-tol <level gcr/etc.>   # The coarse solver tolerance for each level (default 0.25, only for levels 1+)\n");
  printf("    --mg-coarse-solver-maxiter <level n>      # The coarse solver maxiter for each level (default 100)\n");
  printf("    --mg-smoother <level mr/etc.>             # The smoother to use for multigrid (default mr)\n");
  printf("    --mg-smoother-tol <level resid_tol>       # The smoother tolerance to use for each multigrid (default 0.25)\n");
  printf("    --mg-schwarz-type <level false/add/mul>   # Whether to use Schwarz preconditioning (requires MR smoother and GCR setup solver) (default false)\n");
  printf("    --mg-schwarz-cycle <level cycle>          # The number of Schwarz cycles to apply per smoother application (default=1)\n");
  printf("    --mg-block-size <level x y z t>           # Set the geometric block size for the each multigrid level's transfer operator (default 4 4 4 4)\n");
  printf("    --mg-mu-factor <level factor>             # Set the multiplicative factor for the twisted mass mu parameter on each level (default 1)\n");
  printf("    --mg-generate-nullspace <true/false>      # Generate the null-space vector dynamically (default true)\n");
  printf("    --mg-generate-all-levels <true/talse>     # true=generate nul space on all levels, false=generate on level 0 and create other levels from that (default true)\n");
  printf("    --mg-load-vec file                        # Load the vectors \"file\" for the multigrid_test (requires QIO)\n");
  printf("    --mg-save-vec file                        # Save the generated null-space vectors \"file\" from the multigrid_test (requires QIO)\n");
  printf("    --mg-vebosity <level verb>                # The verbosity to use on each level of the multigrid (default silent)\n");
  printf("    --nsrc <n>                                # How many spinors to apply the dslash to simultaneusly (experimental for staggered only)\n");
  printf("    --msrc <n>                                # Used for testing non-square block blas routines where nsrc defines the other dimension\n");


  /////////////////////
  // QKXTM additions //
  /////////////////////


  //-C.K. Generic INPUT
  printf("    --traj                                    # Trajectory of the configuration\n");
  printf("    --csw                                     # Clover csw coefficient (default 1.57551)\n");
  printf("    --load-gauge-smeared                      # Load smeared gauge field \"file\" (in LIME format)\n");
  printf("    --verbosity-level                         # Verbosity level (verbose/summarize/silent, default: summarize)\n");


  //-C.K. Correlation function INPUT
  printf("    --x_source                                # Source position in x direction (default 0)\n");
  printf("    --y_source                                # Source position in y direction (default 0)\n");
  printf("    --z_source                                # Source position in z direction (default 0)\n");
  printf("    --t_source                                # Source position in t direction (default 0)\n");
  printf("    --pathListSinkSource                      # Path to sink-source separations (default \" list_tsinksource.txt \")\n");
  printf("    --pathListRun3pt                          # Path to source positions to run for 2pt- and 3pt- functions (default \" listrun3pt.txt \")\n");
  printf("    --run3pt                                  # Option to choose whether to run for all (=all/ALL) source-positions, for none (=none/NONE)\n"
	 "                                                  or only some (=file/FILE, given in --pathListRun3pt) (default \" all \")\n");
  printf("    --Ntsink                                  # Number of sink-source separations (default \" list_tsinksource.txt \")\n");
  printf("    --Q-sqMax                                 # The maximum Q^2 momentum (loop/correlators) (default 0)\n");
  printf("    --nsmearAPE                               # Number of APE smearing iterations (default 20)\n");
  printf("    --alphaAPE                                # APE smearing parameter (default 0.5)\n");
  printf("    --nsmearGauss                             # Number of Gauss smearing iterations (default 50)\n");
  printf("    --alphaGauss                              # Gauss smearing parameter (default 4.0)\n");
  printf("    --corr-dirname                            # Dir name where to save correlator (default \"./\")\n");
  //printf("    --twop-filename                           # File name to save twopoint function (default \"twop\")\n");
  printf("    --threep-filename                         # File name to save threepoint function (default \"threep\")\n");
  printf("    --prop_path                               # File name to save propagators is (default \"prop_path\")\n");
  printf("    --numSourcePositions                      # The number of source positions we want to calculate (default 1)\n");
  printf("    --pathListSourcePositions                 # Path where the list with the source positions is (default \" listSourcePositions.txt \")\n");
  printf("    --corr-file-format                        # file format for the 2pt-3pt functions, ASCII/HDF5 (default \"ASCII_format\")\n");
  printf("    --check-corr-files                        # check if 2pt-functions exist to avoid reproducing (default \"no\")\n");
  printf("    --proj-list                               # path to a file-list of projectors for 3pt function (default: only G4)\n");
  printf("    --corr-write-space                        # write the correlation functions in position space (MOMENTUM/POSITION, default: MOMENTUM)\n");

  //-C.K. Loop INPUT
  printf("    --Q-sqMax-loop                            # The maximum Q^2 momentum (loop) (default 0)\n");
  printf("    --seed                                    # Seed for ranlux random number generator (default 100)\n");
  printf("    --Nstoch                                  # Number of stochastic noise vectors for loop (default 100)\n");
  printf("    --NdumpStep                               # Every how many noise vectors it will dump the data (default 10)\n");
  printf("    --loop-filename                           # File name to save loops (default \"loop\")\n");
  printf("    --loop-file-format                        # file format for the loops, ASCII/HDF5 (default \"ASCII_format\")\n");
  printf("    --source-type                             # Stochastic source type (unity/random) (default random)\n");
  printf("    --useEven                                 # Whether to use Even-Even operator (yes/no, default no)\n");
#ifdef HAVE_ARPACK
  printf("    --pathEigenVectorsUp                      # Path where the eigenVectors for up flavor are (default ev_u.0000)\n");
  printf("    --pathEigenVectorsDown                    # Path where the eigenVectors for up flavor are (default ev_d.0000)\n");
  printf("    --pathEigenValuesUp                       # Path where the eigenVectors for up flavor are (default evals_u.dat)\n");
  printf("    --pathEigenValuesDown                     # Path where the eigenVectors for up flavor are (default evals_d.dat)\n");

  //-C.K. ARPACK EXACT INPUT
  printf("    --PolyDeg                                 # The degree of the polynomial Acceleration (default 100)\n");
  printf("    --nEv                                     # Number of eigenvalues requested by ARPACK (default 100)\n");
  printf("    --nKv                                     # Total size of the Krylov space used by ARPACK (default 200)\n");
  printf("    --spectrumPart                            # Which part of the spectrum we need (Options: SR,LR,SM,LM,SI,LI, default SR)\n");
  printf("    --isACC                                   # Whether we want to use polynomial acceleration (yes/no, default yes)\n");
  printf("    --tolARPACK                               # Tolerance for convergence, used by ARPACK (default 1.0e-5)\n");
  printf("    --maxIterARPACK                           # Maximum iterations number for ARPACK (default 100000)\n");
  printf("    --modeARPACK                              # MODE for ARPACK soluton (default 1)\n");
  printf("    --pathArpackLogfile                       # Path to the ARPACK log file (default  \"arpack.log\")\n");
  printf("    --aminARPACK                              # amin parameter used in Cheb. Poly. Acc. (default 3.0e-4)\n");
  printf("    --amaxARPACK                              # amax parameter used in Cheb. Poly. Acc. (default 3.5)\n");
  printf("    --useFullOp                               # Whether to use the Full Operator (yes,no, default no)\n");
  printf("    --defl-steps <steps>                      # Number of deflation steps (default: 1, the total requested NeV)\n");
  printf("    --defl-step-NeV <step> <NeV_at_step>      # Number of eigenvectors to deflate at step <step> (default: the total requested NeV)\n");
#endif
  printf("    --k-probing <n>                           # Hierarchical probing, where neighbors distance D=2**k (default 0: No probing)\n");
  printf("    --spinColorDil <true/false>               # Whether we want spin color dilution (default false)\n");
  printf("    --hadamLow <n>                            # From which Hadamard vector to start (default 0)\n");
  printf("    --hadamHigh <n>                           # Up to which Hadamard vector to stop  (default max, max will be set later in the code, max not included in the for loop)\n");
  
  //--------//

  printf("    --help                                    # Print out this message\n"); 
  usage_extra(argv); 
#ifdef MULTI_GPU
  char msg[]="multi";
#else
  char msg[]="single";
#endif  
  printf("Note: this program is %s GPU build\n", msg);
  exit(1);
  return ;
}

static int process_command_line_option(int argc, char** argv, int* idx)
{
#ifdef MULTI_GPU
  char msg[]="multi";
#else
  char msg[]="single";
#endif

  int ret = -1;
  
  int i = *idx;

  if( strcmp(argv[i], "--help")== 0){
    usage(argv);
  }

  if( strcmp(argv[i], "--verify") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    

    if (strcmp(argv[i+1], "true") == 0){
      verify_results = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      verify_results = false;
    }else{
      fprintf(stderr, "ERROR: invalid verify type\n");	
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--device") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    device = atoi(argv[i+1]);
    if (device < 0 || device > 16){
      printf("ERROR: Invalid CUDA device number (%d)\n", device);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--prec") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    prec =  get_prec(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--prec-sloppy") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    prec_sloppy =  get_prec(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--prec-precondition") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    prec_precondition =  get_prec(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--prec-null") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    prec_null =  get_prec(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--recon") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    link_recon =  get_recon(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--recon-sloppy") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    link_recon_sloppy =  get_recon(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--recon-precondition") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    link_recon_precondition =  get_recon(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--dim") == 0){
    if (i+4 >= argc){
      usage(argv);
    }
    xdim= atoi(argv[i+1]);
    if (xdim < 0 || xdim > 512){
      printf("ERROR: invalid X dimension (%d)\n", xdim);
      usage(argv);
    }
    i++;

    if (i+1 >= argc){
      usage(argv);
    }
    ydim= atoi(argv[i+1]);
    if (ydim < 0 || ydim > 512){
      printf("ERROR: invalid Y dimension (%d)\n", ydim);
      usage(argv);
    }
    i++;

    if (i+1 >= argc){
      usage(argv);
    }
    zdim= atoi(argv[i+1]);
    if (zdim < 0 || zdim > 512){
      printf("ERROR: invalid Z dimension (%d)\n", zdim);
      usage(argv);
    }
    i++;

    if (i+1 >= argc){
      usage(argv);
    }
    tdim= atoi(argv[i+1]);
    if (tdim < 0 || tdim > 512){
      printf("ERROR: invalid T dimension (%d)\n", tdim);
      usage(argv);
    }
    i++;

    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--xdim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    xdim= atoi(argv[i+1]);
    if (xdim < 0 || xdim > 512){
      printf("ERROR: invalid X dimension (%d)\n", xdim);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--ydim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    ydim= atoi(argv[i+1]);
    if (ydim < 0 || ydim > 512){
      printf("ERROR: invalid T dimension (%d)\n", ydim);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }


  if( strcmp(argv[i], "--zdim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    zdim= atoi(argv[i+1]);
    if (zdim < 0 || zdim > 512){
      printf("ERROR: invalid T dimension (%d)\n", zdim);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--tdim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    tdim =  atoi(argv[i+1]);
    if (tdim < 0 || tdim > 512){
      printf("Error: invalid t dimension");
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--sdim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    int sdim =  atoi(argv[i+1]);
    if (sdim < 0 || sdim > 512){
      printf("ERROR: invalid S dimension\n");
      usage(argv);
    }
    xdim=ydim=zdim=sdim;
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--Lsdim") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    int Ls =  atoi(argv[i+1]);
    if (Ls < 0 || Ls > 128){
      printf("ERROR: invalid Ls dimension\n");
      usage(argv);
    }
    Lsdim=Ls;
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--dagger") == 0){
    dagger = QUDA_DAG_YES;
    ret = 0;
    goto out;
  }	
  
  if( strcmp(argv[i], "--partition") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
#ifdef MULTI_GPU
    int value  =  atoi(argv[i+1]);
    for(int j=0; j < 4;j++){
      if (value &  (1 << j)){
	commDimPartitionedSet(j);
	dim_partitioned[j] = 1;
      }
    }
#else
    printfQuda("WARNING: Ignoring --partition option since this is a single-GPU build.\n");
#endif
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--kernel-pack-t") == 0){
    kernel_pack_t = true;
    ret= 0;
    goto out;
  }


  if( strcmp(argv[i], "--multishift") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    

    if (strcmp(argv[i+1], "true") == 0){
      multishift = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      multishift = false;
    }else{
      fprintf(stderr, "ERROR: invalid multishift boolean\n");	
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--gridsize") == 0){
    if (i+4 >= argc){
      usage(argv);
    }     
    int xsize =  atoi(argv[i+1]);
    if (xsize <= 0 ){
      printf("ERROR: invalid X grid size");
      usage(argv);
    }
    gridsize_from_cmdline[0] = xsize;
    i++;

    int ysize =  atoi(argv[i+1]);
    if (ysize <= 0 ){
      printf("ERROR: invalid Y grid size");
      usage(argv);
    }
    gridsize_from_cmdline[1] = ysize;
    i++;

    int zsize =  atoi(argv[i+1]);
    if (zsize <= 0 ){
      printf("ERROR: invalid Z grid size");
      usage(argv);
    }
    gridsize_from_cmdline[2] = zsize;
    i++;

    int tsize =  atoi(argv[i+1]);
    if (tsize <= 0 ){
      printf("ERROR: invalid T grid size");
      usage(argv);
    }
    gridsize_from_cmdline[3] = tsize;
    i++;

    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--xgridsize") == 0){
    if (i+1 >= argc){ 
      usage(argv);
    }     
    int xsize =  atoi(argv[i+1]);
    if (xsize <= 0 ){
      printf("ERROR: invalid X grid size");
      usage(argv);
    }
    gridsize_from_cmdline[0] = xsize;
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--ygridsize") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    int ysize =  atoi(argv[i+1]);
    if (ysize <= 0 ){
      printf("ERROR: invalid Y grid size");
      usage(argv);
    }
    gridsize_from_cmdline[1] = ysize;
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--zgridsize") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    int zsize =  atoi(argv[i+1]);
    if (zsize <= 0 ){
      printf("ERROR: invalid Z grid size");
      usage(argv);
    }
    gridsize_from_cmdline[2] = zsize;
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--tgridsize") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    int tsize =  atoi(argv[i+1]);
    if (tsize <= 0 ){
      printf("ERROR: invalid T grid size");
      usage(argv);
    }
    gridsize_from_cmdline[3] = tsize;
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--dslash-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    dslash_type =  get_dslash_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--flavor") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    twist_flavor =  get_flavor_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--inv-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    inv_type = get_solver_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--precon-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    precon_type = get_solver_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mass") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    mass = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--kappa") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    kappa = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--compute-clover") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    if (strcmp(argv[i+1], "true") == 0){
      compute_clover = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      compute_clover = false;
    }else{
      fprintf(stderr, "ERROR: invalid compute_clover type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--clover-coeff") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    clover_coeff = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mu") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    mu = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--anisotropy") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    anisotropy = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--tol") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    tol= atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--tolhq") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    tol_hq= atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mass-normalization") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    normalization = get_mass_normalization_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--matpc") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    matpc_type = get_matpc_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--solve-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    solve_type = get_solve_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--load-gauge") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(latfile, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--nsrc") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    Nsrc = atoi(argv[i+1]);
    if (Nsrc < 1 || Nsrc > 128){
      printf("ERROR: invalid number of sources (Nsrc=%d)\n", Nsrc);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--msrc") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    Msrc = atoi(argv[i+1]);
    if (Msrc < 1 || Msrc > 128){
      printf("ERROR: invalid number of sources (Msrc=%d)\n", Msrc);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--test") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    test_type = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;	    
  }
    
  if( strcmp(argv[i], "--mg-nvec") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    nvec[level] = atoi(argv[i+1]);
    if (nvec[level] < 0 || nvec[level] > 128){
      printf("ERROR: invalid number of vectors (%d)\n", nvec[level]);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-levels") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    mg_levels= atoi(argv[i+1]);
    if (mg_levels < 2 || mg_levels > QUDA_MAX_MG_LEVEL){
      printf("ERROR: invalid number of multigrid levels (%d)\n", mg_levels);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-nu-pre") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    nu_pre= atoi(argv[i+1]);
    if (nu_pre < 0 || nu_pre > 20){
      printf("ERROR: invalid pre-smoother applications value (nu_pre=%d)\n", nu_pre);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-nu-post") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    nu_post= atoi(argv[i+1]);
    if (nu_post < 0 || nu_post > 20){
      printf("ERROR: invalid pre-smoother applications value (nu_pist=%d)\n", nu_post);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-inv") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    setup_inv[level] = get_solver_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-iters") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    num_setup_iter[level] = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-tol") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    setup_tol = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-setup-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    if( strcmp(argv[i+1], "test") == 0)
      setup_type = QUDA_TEST_VECTOR_SETUP;
    else if( strcmp(argv[i+1], "null")==0)
      setup_type = QUDA_NULL_VECTOR_SETUP;
    else {
      fprintf(stderr, "ERROR: invalid setup type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-pre-orth") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    if (strcmp(argv[i+1], "true") == 0){
      pre_orthonormalize = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      pre_orthonormalize = false;
    }else{
      fprintf(stderr, "ERROR: invalid pre orthogonalize type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-post-orth") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    if (strcmp(argv[i+1], "true") == 0){
      post_orthonormalize = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      post_orthonormalize = false;
    }else{
      fprintf(stderr, "ERROR: invalid post orthogonalize type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-omega") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    omega = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-verbosity") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    mg_verbosity[level] = get_verbosity_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-coarse-solver") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 1 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d for coarse solver", level);
      usage(argv);
    }
    i++;

    coarse_solver[level] = get_solver_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-smoother") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    smoother_type[level] = get_solver_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-smoother-tol") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    smoother_tol[level] = atof(argv[i+1]);

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-coarse-solver-tol") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 1 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d for coarse solver", level);
      usage(argv);
    }
    i++;

    coarse_solver_tol[level] = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-coarse-solver-maxiter") == 0){
    if (i+2 >= argc){
      usage(argv);
    }

    int level = atoi(argv[i+1]);
    if (level < 1 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d for coarse solver", level);
      usage(argv);
    }
    i++;

    coarse_solver_maxiter[level] = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }


  if( strcmp(argv[i], "--mg-schwarz-type") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    schwarz_type[level] = get_schwarz_type(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-schwarz-cycle") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    schwarz_cycle[level] = atoi(argv[i+1]);
    if (schwarz_cycle[level] < 0 || schwarz_cycle[level] >= 128) {
      printf("ERROR: invalid Schwarz cycle value requested %d for level %d",
	     level, schwarz_cycle[level]);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-block-size") == 0){
    if (i+5 >= argc){
      usage(argv);
    }     
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    int xsize =  atoi(argv[i+1]);
    if (xsize <= 0 ){
      printf("ERROR: invalid X block size");
      usage(argv);
    }
    geo_block_size[level][0] = xsize;
    i++;

    int ysize =  atoi(argv[i+1]);
    if (ysize <= 0 ){
      printf("ERROR: invalid Y block size");
      usage(argv);
    }
    geo_block_size[level][1] = ysize;
    i++;

    int zsize =  atoi(argv[i+1]);
    if (zsize <= 0 ){
      printf("ERROR: invalid Z block size");
      usage(argv);
    }
    geo_block_size[level][2] = zsize;
    i++;

    int tsize =  atoi(argv[i+1]);
    if (tsize <= 0 ){
      printf("ERROR: invalid T block size");
      usage(argv);
    }
    geo_block_size[level][3] = tsize;
    i++;

    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-mu-factor") == 0){
    if (i+2 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level >= QUDA_MAX_MG_LEVEL) {
      printf("ERROR: invalid multigrid level %d", level);
      usage(argv);
    }
    i++;

    double factor =  atof(argv[i+1]);
    mu_factor[level] = factor;
    i++;
    ret=0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-generate-nullspace") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    if (strcmp(argv[i+1], "true") == 0){
      generate_nullspace = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      generate_nullspace = false;
    }else{
      fprintf(stderr, "ERROR: invalid generate nullspace type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-generate-all-levels") == 0){
    if (i+1 >= argc){
      usage(argv);
    }

    if (strcmp(argv[i+1], "true") == 0){
      generate_all_levels = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      generate_all_levels = false;
    }else{
      fprintf(stderr, "ERROR: invalid value for generate_all_levels type\n");
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-load-vec") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    strcpy(vec_infile, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--mg-save-vec") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    strcpy(vec_outfile, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--niter") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    niter= atoi(argv[i+1]);
    if (niter < 1 || niter > 1e6){
      printf("ERROR: invalid number of iterations (%d)\n", niter);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--ngcrkrylov") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    gcrNkrylov = atoi(argv[i+1]);
    if (gcrNkrylov < 1 || gcrNkrylov > 1e6){
      printf("ERROR: invalid number of gcrkrylov iterations (%d)\n", gcrNkrylov);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--pipeline") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    pipeline = atoi(argv[i+1]);
    if (pipeline < 0 || pipeline > 8){
      printf("ERROR: invalid pipeline length (%d)\n", pipeline);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--solution-pipeline") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    solution_accumulator_pipeline = atoi(argv[i+1]);
    if (solution_accumulator_pipeline < 0 || solution_accumulator_pipeline > 16){
      printf("ERROR: invalid solution pipeline length (%d)\n", solution_accumulator_pipeline);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }


  /////////////////////
  // QKXTM additions //
  /////////////////////

  //-C.K. Generic INPUT
  if( strcmp(argv[i], "--traj") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    traj = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }


  if( strcmp(argv[i], "--load-gauge-smeared") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(latfile_smeared, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--verbosity-level") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(verbosity_level, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--csw") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    csw = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  //-C.K. Correlation functions INPUT
  if( strcmp(argv[i], "--x-source") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    src[0] = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--y-source") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    src[1] = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--z-source") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    src[2] = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--t-source") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    src[3] = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--Ntsink") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    Ntsink = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--pathListSinkSource") == 0){
    if(i+1 >= argc){
      usage(argv);
    }
    strcpy(pathList_tsink,argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--Q-sqMax") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    Q_sq = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--nsmearAPE") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    nsmearAPE = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--alphaAPE") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    alphaAPE = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--nsmearGauss") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    nsmearGauss = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--alphaGauss") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    alphaGauss = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--corr-dirname") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(corr_dirname, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  /*
  if( strcmp(argv[i], "--twop-filename") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(twop_filename, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  */
  
  if( strcmp(argv[i], "--prop-path") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(prop_path, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--threep-filename") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(threep_filename, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--numSourcePositions") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    numSourcePositions = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--pathListSourcePositions") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(pathListSourcePositions, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--pathListRun3pt") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(pathListRun3pt, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--run3pt") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(run3pt, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--corr-file-format") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    corr_file_format = strdup(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--check-corr-files") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    strcpy(check_file_exist, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--proj-list") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(proj_list_file, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--corr-write-space") ==0){
    if(i+1 >= argc){
      usage(argv);
    }    
    corr_write_space = strdup(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--seed") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    seed =(unsigned long int) atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--Nstoch") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    Nstoch = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--NdumpStep") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    Ndump = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--loop-filename") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(loop_fname, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--loop-file-format") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    loop_file_format = strdup(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--source-type") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    strcpy(source_type, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  
  if( strcmp(argv[i], "--useEven") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    if (strcmp(argv[i+1], "true") == 0){
      isEven = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      isEven = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for useEven (true/false)\n");
      exit(1);
    }
    
    i++;
    ret = 0;
    goto out;
  }
#ifdef HAVE_ARPACK
  //-Loop info with ARPACK enabled

  if( strcmp(argv[i], "--defl-steps") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    defl_steps = atoi(argv[i+1]);
    if (defl_steps < 0 || defl_steps > 10){
      printf("ERROR: invalid number of defl steps (%d): \n", defl_steps);
      usage(argv);
    }
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--defl-step-NeV") == 0){
    if (i+1 >= argc){
      usage(argv);
    }
    int level = atoi(argv[i+1]);
    if (level < 0 || level > 10) {
      printf("ERROR: invalid defl step level %d", level);
      usage(argv);
    }
    i++;

    defl_step_nEv[level] = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--pathEigenVectorsUp") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(pathEigenVectorsUp, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--pathEigenVectorsDown") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(pathEigenVectorsDown, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--pathEigenValuesUp") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(pathEigenValuesUp, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--pathEigenValuesDown") == 0){
    if (i+1 >= argc){
      usage(argv);
    }     
    strcpy(pathEigenValuesDown, argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  //-C.K. ARPACK INPUT
  if( strcmp(argv[i], "--PolyDeg") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    PolyDeg = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--nEv") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    nEv = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  
  if( strcmp(argv[i], "--nKv") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    nKv = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
  

  if( strcmp(argv[i], "--spectrumPart") ==0){
    if(i+1 >= argc){
      usage(argv);
    }    
    spectrumPart = strdup(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--isACC") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    if (strcmp(argv[i+1], "true") == 0){
      isACC = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      isACC = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for issACC (true/false)\n");
      exit(1);
    }
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--tolARPACK") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    tolArpack = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--maxIterARPACK") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    maxIterArpack = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--modeARPACK") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    modeArpack = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--pathArpackLogfile") == 0){
    if(i+1 >= argc){
      usage(argv);
    }
    strcpy(arpack_logfile,argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
 
  if( strcmp(argv[i], "--aminARPACK") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    amin = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }


  if( strcmp(argv[i], "--amaxARPACK") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    amax = atof(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }
#endif
 
  if( strcmp(argv[i], "--useFullOp") ==0){
    if(i+1 >= argc){
      usage(argv);
    }
    if (strcmp(argv[i+1], "true") == 0){
      isFullOp = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      isFullOp = false;
    } else {
      fprintf(stderr, "ERROR: invalid value for useFullOp (true/false)\n");
      exit(1);
    }
    
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--k-probing") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    k_probing = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--hadamLow") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    hadamLow = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--hadamHigh") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    
    hadamHigh = atoi(argv[i+1]);
    i++;
    ret = 0;
    goto out;
  }

  if( strcmp(argv[i], "--spinColorDil") == 0){
    if (i+1 >= argc){
      usage(argv);
    }	    

    if (strcmp(argv[i+1], "true") == 0){
      spinColorDil = true;
    }else if (strcmp(argv[i+1], "false") == 0){
      spinColorDil = false;
    }else{
      fprintf(stderr, "ERROR: invalid spinColorDil type\n");	
      exit(1);
    }

    i++;
    ret = 0;
    goto out;
  }

  //-----//

  if( strcmp(argv[i], "--version") == 0){
    printf("This program is linked with QUDA library, version %s,", 
	   get_quda_ver_str());
    printf(" %s GPU build\n", msg);
    exit(0);
  }
  
 out:
  *idx = i;
  return ret ;  
}

void read_command_line(int argc, char **argv, plegma::PLEGMA_params *params) {
  set_default_values();
  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }
    printf("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }
  finalize_values();
  set_PLEGMA_params(params);
}
