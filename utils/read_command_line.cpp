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
//QudaPrecision prec_refinement_sloppy = QUDA_INVALID_PRECISION;
QudaPrecision  prec_precondition = QUDA_INVALID_PRECISION;
QudaPrecision prec_null = QUDA_INVALID_PRECISION;
int xdim = 24;
int ydim = 24;
int zdim = 24;
int tdim = 24;
int Lsdim = 16;
QudaDagType dagger = QUDA_DAG_NO;
int gridsize_from_cmdline[4] = {1,1,1,1};
QudaDslashType dslash_type = QUDA_TWISTED_CLOVER_DSLASH;
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
double tol = 1e-9; // KH: I switch from 1e-7 to 1e-9
double tol_hq = 0.1;
double reliable_delta = 0.1; // KH: Why reliable delta was missing?
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

//double setup_tol[QUDA_MAX_MG_LEVEL] = { };
//int setup_maxiter[QUDA_MAX_MG_LEVEL] = { };
//int setup_maxiter_refresh[QUDA_MAX_MG_LEVEL] = { };

double setup_tol = 5e-6;
QudaSetupType setup_type = QUDA_NULL_VECTOR_SETUP;//
bool pre_orthonormalize = false;//
bool post_orthonormalize = true;//
double omega = 0.85;
//QudaSolveType coarse_solve_type[QUDA_MAX_MG_LEVEL] = { };
//QudaSolveType smoother_solve_type[QUDA_MAX_MG_LEVEL] = { };
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

// function call as last after all params have been read
static void finalize_values(){ // Do we need this?
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec; // why the sloppy precision is the same as the normal precision?
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

static void printBasicOptions(){
  _PRINT_("dims, %d %d %d %d\n",xdim,ydim,zdim,tdim);
  _PRINT_("gridsize, %d %d %d %d\n",gridsize_from_cmdline[0],gridsize_from_cmdline[1],gridsize_from_cmdline[2],gridsize_from_cmdline[3]);
  _PRINT_("load-gauge, %s\n", latfile);
}

static void printBasicOptionsWsolver(){
  _PRINT_("Q-prec, %d\n",prec);
  _PRINT_("Q-prec-sloppy, %d\n",prec_sloppy);
  _PRINT_("Q-prec-precondition, %d\n",prec_precondition);
  _PRINT_("Q-recon, %d\n",link_recon);
  _PRINT_("Q-recon-sloppy, %d\n",link_recon_sloppy);
  _PRINT_("Q-recon-precondition, %d\n",link_recon_precondition);
  _PRINT_("Q-dslash-type, %d\n", dslash_type);
  _PRINT_("Q-dagger, %d\n", dagger);
  _PRINT_("Q-kernel-pack-t, %d\n",kernel_pack_t);
  _PRINT_("Q-flavor, %d\n",twist_flavor);
  _PRINT_("Q-niter, %d\n",niter);
  _PRINT_("Q-ngcrkrylov, %d\n",gcrNkrylov);
  _PRINT_("Q-pipeline, %d\n",pipeline);
  _PRINT_("Q-solution-pipeline, %d\n",solution_accumulator_pipeline);
  _PRINT_("Q-inv-type, %d\n",inv_type);
  _PRINT_("Q-precon-type, %d\n",precon_type);
  _PRINT_("Q-kappa, %f\n",kappa);
  _PRINT_("Q-mu, %f\n",mu);
  _PRINT_("Q-csw, %f\n",csw);
  _PRINT_("Q-mass-normalization, %d\n",normalization);
  _PRINT_("Q-matpc, %d\n",matpc_type);
  _PRINT_("Q-solve-type, %d\n", solve_type);
  _PRINT_("Q-tol, %+e\n", tol);
  _PRINT_("Q-tolhq, %+e\n", tol_hq);
  _PRINT_("Q-reliable-delta, %+f\n", reliable_delta);
  _PRINT_("Q-mg-levels, %d\n", mg_levels);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-nvec, %d %d\n",i,nvec[i]);
  _PRINT_("Q-mg-nu-pre, %d\n", nu_pre);
  _PRINT_("Q-mg-nu-post, %d\n", nu_post);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-setup-inv, %d %d\n",i,setup_inv[i]);
  _PRINT_("Q-mg-setup-tol, %+e\n", setup_tol);
  _PRINT_("Q-mg-omega, %f\n", omega);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-smoother, %d %d\n", i, smoother_type[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-block-size, %d \t %d %d %d %d\n", i, geo_block_size[i][0], geo_block_size[i][1], geo_block_size[i][2], geo_block_size[i][3]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-mu-factor, %d %f\n", i, mu_factor[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-verbosity, %d %d\n", i, mg_verbosity[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-coarse-solver, %d %d\n", i, coarse_solver[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-coarse-solver-tol, %d %f\n",i,coarse_solver_tol[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-coarse-solver-maxiter, %d %d\n", i, coarse_solver_maxiter[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-schwarz-type, %d %d\n", i, schwarz_type[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-schwarz-cycle, %d %d\n", i, schwarz_cycle[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-smoother-tol, %d %f\n",i,smoother_tol[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-setup-iters, %d %d\n", i, num_setup_iter[i]);
  _PRINT_("Q-mg-pre-orth, %d\n", pre_orthonormalize);
  _PRINT_("Q-mg-post-orth, %d\n", post_orthonormalize);
}

void basicOptions(Options &opt, plegma::PLEGMA_params *params, bool showThem){
  bool isFound;
  
  opt.setForced("dims","Set dimensions (X Y Z T), default (24 24 24 24)", false, xdim, ydim, zdim, tdim);
  if(!opt.getIsHelp()) if( (xdim < 0 || xdim > 512) || (ydim < 0 || ydim > 512)  || (zdim < 0 || zdim > 512) || (tdim < 0 || tdim > 512)) _ERROR_("Error: dims should be > 0 and < 512\n");

  opt.setForced("gridsize","Set grid size (X Y Z T), default (1 1 1 1)", false,gridsize_from_cmdline[0], gridsize_from_cmdline[1],gridsize_from_cmdline[2],gridsize_from_cmdline[3]);
  if(!opt.getIsHelp()) for(int i = 0 ; i < 4; i++) if(gridsize_from_cmdline[i]<=0) _ERROR_("Error: Negative gridsize in %d dim\n",i);

  std::string gfile;
  isFound=opt.set("load-gauge", "Path to the gauge field, default (empty string)", false, gfile);
  if(isFound)strcpy(latfile,gfile.c_str());

  if(showThem && !opt.getIsHelp()) printBasicOptions();
  
  params->lL[0] = xdim;
  params->lL[1] = ydim;
  params->lL[2] = zdim;
  params->lL[3] = tdim;
  for(int i=0; i<4; i++)
    params->procs[i] = gridsize_from_cmdline[i];

  // !! remove later from plegma params
  params->Nsources = 0;
  params->Q_sq = 0;
}

template<typename T> static inline void map_to_array_MG(std::map<int,T> &tpl, T *arr, T def, T noSm, T noBig, std::string err){
  for(int i=0; i<QUDA_MAX_MG_LEVEL; i++) arr[i] = def;
  if(tpl.empty()) return;
  typename std::map<int,T>::iterator it = tpl.begin();
  while(it != tpl.end()){
    int lvl=it->first;
    T val = it->second;
    if(lvl < 0 || lvl >= QUDA_MAX_MG_LEVEL) _ERROR_("ERROR: invalid multigrid level %d", lvl);
    if(val < noSm || val > noBig) _ERROR_(err.c_str());
    arr[lvl]=val;
    it++;
  }
}

template<typename T> static inline void map_to_array_MG(std::map<int,std::string> &tpl, T *arr, T def, std::function<T(const char*)> func){
  for(int i=0; i<QUDA_MAX_MG_LEVEL; i++) arr[i] = def;
  if(tpl.empty()) return;
  typename std::map<int,std::string>::iterator it = tpl.begin();
  while(it != tpl.end()){
    int lvl=it->first;
    T val = func((it->second).c_str());
    if(lvl < 0 || lvl >= QUDA_MAX_MG_LEVEL) _ERROR_("ERROR: invalid multigrid level %d", lvl);
    arr[lvl]=val;
    it++;
  }
}

void basicOptionsWsolver(Options &opt, plegma::PLEGMA_params *params, bool showThem){
  basicOptions(opt,params,showThem);
  set_default_values(); // default values for MG
  bool isFound;
  std::string tmpString;
  bool tmpBool;
    
  isFound=opt.set("Q-prec", "Precision in the GPU, options (double,single,half), default (single)", false, tmpString);
  if(isFound)prec = get_prec(tmpString.c_str());

  isFound=opt.set("Q-prec-sloppy", "Sloppy precision in the GPU, options (double,single,half), default (invalid)", false, tmpString);
  if(isFound)prec_sloppy = get_prec(tmpString.c_str());


  isFound=opt.set("Q-prec-precondition", "Preconditioner precision in the GPU, options (double,single,half),default (invalid)", false, tmpString);
  if(isFound)prec_precondition = get_prec(tmpString.c_str());

  isFound=opt.set("Q-recon", "Type of link reconstruction, options (8,9,12,13,18), default (18 no reconstruction)", false, tmpString);
  if(isFound)link_recon  = get_recon(tmpString.c_str());

  isFound=opt.set("Q-recon-sloppy", "Type of link reconstruction for sloppy, options (8,9,12,13,18), default (invalid)", false, tmpString);
  if(isFound)link_recon_sloppy  = get_recon(tmpString.c_str());

  isFound=opt.set("Q-recon-precondition", "Type of link reconstruction for precon, options (8,9,12,13,18), default (invalid)", false, tmpString);
  if(isFound)link_recon_precondition  = get_recon(tmpString.c_str());

  opt.setForced("Q-dslash-type", "Set the dslash type, options for now (twisted-mass/twisted-clover)",false, tmpString);
  if(!opt.getIsHelp()){
    if((tmpString != "twisted-mass") && (tmpString != "twisted-clover"))_ERROR_("Error: only twisted-mass or twisted-clover are allowed for now");
    dslash_type =  get_dslash_type(tmpString.c_str());
  }

  isFound=opt.set("Q-dagger", "In case you want the dagger operator, default (false)", false, tmpBool);
  if(isFound && tmpBool) dagger = QUDA_DAG_YES;

  isFound=opt.set("Q-kernel-pack-t", "Kernel packing in the T direction, default (false)", false, tmpBool);
  if(isFound && tmpBool) kernel_pack_t = true;

  isFound=opt.set("Q-flavor", "Twisted mass type of flavor, options for now (singlet), default (singlet)", false,tmpString);
  if(isFound) twist_flavor = get_flavor_type(tmpString.c_str());

  opt.set("Q-niter", "Maximum number of iterations for the solvers, options (1,...,1e6), default (10)",false,niter);
  if(niter<1 || niter>1e6) _ERROR_("Error: Invalid number [%d] for max number of iterations\n",niter);

  opt.set("Q-ngcrkrylov", "The number of inner iterations to use for GCR or BiCGstab-l, options (1,...,1e6), (default 10)",false,gcrNkrylov);
  if(gcrNkrylov<1 || gcrNkrylov>1e6) _ERROR_("Error: Invalid number [%d] for gcrNkrylov iterations\n",gcrNkrylov);

  opt.set("Q-pipeline", "The pipeline length for fused operations in GCR or BiCGstab-l, options(0,...,8), (default 0, no pipelining)", false, pipeline);
  if(pipeline < 0 || pipeline > 8) _ERROR_("Error: Invalid number [%d] for pipeline length\n",pipeline);

  opt.set("Q-solution-pipeline", "The pipeline length for fused solution accumulation, options (0,..,16), (default 0, no pipelining)", false, solution_accumulator_pipeline);
  if (solution_accumulator_pipeline < 0 || solution_accumulator_pipeline > 16)  _ERROR_("Error: Invalid number [%d] for solution pipeline length\n",solution_accumulator_pipeline);

  isFound=opt.set("Q-inv-type", "The type of solver to use, options (cg,bicgstab,gcr), default (cg)", false, tmpString);
  if(isFound) inv_type = get_solver_type(tmpString.c_str());

  isFound=opt.set("Q-precon-type", "The type of precon solver to use, options (mr,none), default (none)", false, tmpString);
  if(isFound)precon_type = get_solver_type(tmpString.c_str());

  opt.set("Q-kappa", "Kappa value of the Dirac operator", false, kappa);
  opt.set("Q-mu", "Twisted mass value", false, mu);
  opt.set("Q-csw", "The coefficient of the clover term", false, csw);

  isFound=opt.set("Q-mass-normalization", "Normalization of the dirac operator,options (kappa,mass,asym-mass), default (kappa)", false, tmpString);
  if(isFound) normalization = get_mass_normalization_type(tmpString.c_str());

  isFound=opt.set("Q-matpc", "Operator preconditioning type, options (even-even, odd-odd, even-even-asym, odd-odd-asym), default(even-even)", false, tmpString);
  if(isFound) matpc_type = get_matpc_type(tmpString.c_str());

  isFound=opt.set("Q-solve-type", "The way to solve the system, options (direct, direct-pc, normop, normop-pc, normerr, normerr-pc), default (direct-pc)", false, tmpString);
  if(isFound) solve_type = get_solve_type(tmpString.c_str());

  opt.set("Q-tol", "The L2 residual tolerance, default (1e-09)", false, tol);
  opt.set("Q-tolhq", "Set heavy-quark residual tolerance, default (0.1)", false, tol_hq);
  opt.set("Q-reliable-delta", "The delta factor for the reliable updates, default (0.1)", false, reliable_delta);

  //=================================== Multigrid related =======================//
  opt.set("Q-mg-levels", "The number of multigrid levels to do. One level has no meaning. (default 2)", false, mg_levels);
  
  std::map<int,int> tpl_int_int;
  std::map<int,std::string> tpl_int_string;
  std::map<int,double> tpl_int_double;

  isFound=opt.set("Q-mg-nvec", "Number of null-space vectors for multigrid, usage (level,nvec)", false, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, nvec, 24, 1, 128, "ERROR: invalid number of vectors");

  isFound=opt.set("Q-mg-nu-pre", "Number of pre-smoother applications, 0-20", false, nu_pre);
  if(isFound) if (nu_pre < 0 || nu_pre > 20) _ERROR_("ERROR: invalid pre-smoother applications value (nu_pre=%d)\n", nu_pre);

  isFound=opt.set("Q-mg-nu-post", "Number of post-smoother applications, 0-20", false, nu_post);
  if(isFound) if (nu_post < 0 || nu_post > 20) _ERROR_("ERROR: invalid post-smoother applications value (nu_post=%d)\n", nu_post);
  
  tpl_int_string.clear();
  isFound=opt.set("Q-mg-setup-inv", "The inverter to use for the setup of multigrid, usage(level,inv), (default bicgstab)", false, tpl_int_string);
  map_to_array_MG<QudaInverterType>(tpl_int_string, setup_inv, QUDA_BICGSTAB_INVERTER, get_solver_type);

  opt.set("Q-mg-setup-tol", "The tolerance to use for the setup of multigrid, (default 5e-6)", false, setup_tol);
  opt.set("Q-mg-omega", "The over/under relaxation factor for the smoother of multigrid (default 0.85)", false, omega);

  tpl_int_string.clear();
  isFound=opt.set("Q-mg-smoother", "The smoother to use for multigrid, usage(level,inv), (default mr)", false, tpl_int_string);
  map_to_array_MG<QudaInverterType>(tpl_int_string, smoother_type, QUDA_MR_INVERTER, get_solver_type);

  std::vector<int> intVec;
  isFound=opt.set("Q-mg-block-size", "Set the geometric block size for the each multigrid level's transfer operator (default 4 4 4 4)", false, intVec);
  if(isFound){
    if(intVec.size()%5 != 0) _ERROR_("Error: For Q-mg-block-size format is (lvl,X,Y,Z,T)\n");
    int nl=intVec.size()/5;
    if(mg_levels != (nl+1)) _ERROR_("Error: Check that Q-mg-levels agrees with the number of levels provided in the blocks\n");
    for(int i = 0; i < nl; i++)
      for(int j = 0; j < 4; j++)
	geo_block_size[intVec[i*5]][j]=intVec[i*5+1+j];
  }

  tpl_int_double.clear();
  isFound=opt.set("Q-mg-mu-factor", "Set the multiplicative factor for the twisted mass mu parameter on each level, usage(level,float), (default 1)", false, tpl_int_double);
  map_to_array_MG<double>(tpl_int_double, mu_factor, 1, 0, 100, "ERROR: invalid mu factor for the multigrid");

  tpl_int_string.clear();
  isFound=opt.set("Q-mg-verbosity", "The verbosity to use on each level of the multigrid, usage(level, verb), (default summarize)", false, tpl_int_string);
  map_to_array_MG<QudaVerbosity>(tpl_int_string, mg_verbosity, QUDA_SUMMARIZE, get_verbosity_type);
    
  //==============================================//
  // KH: From now on does not exist in the QUDA 0.9. Do we need them?
  tpl_int_string.clear();
  isFound=opt.set("Q-mg-coarse-solver", "The solver to wrap the V cycle on each level, usage(level,inv), (default gcr, only for levels 1+)", false, tpl_int_string);
  map_to_array_MG<QudaInverterType>(tpl_int_string, coarse_solver, QUDA_GCR_INVERTER, get_solver_type);

  tpl_int_double.clear();
  isFound=opt.set("Q-mg-coarse-solver-tol", "The coarse solver tolerance for each level, usage(level,float(0,1)), (default 0.25, only for levels 1+)", false, tpl_int_double);
  map_to_array_MG<double>(tpl_int_double, coarse_solver_tol, 0.25, 0, 1, "ERROR: invalid tolerance for the MG coarse solver");

  tpl_int_int.clear();
  isFound=opt.set("Q-mg-coarse-solver-maxiter", "The coarse solver maxiter for each level, usage(level,int), (default 100)", false, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, coarse_solver_maxiter, 100, 0, 10000, "ERROR: invalid max number of MG setup coarse solver max iter");

  tpl_int_string.clear();
  isFound=opt.set("Q-mg-schwarz-type", "Whether to use Schwarz preconditioning (requires MR smoother and GCR setup solver), usage(level,add/mul) (default false)", false,tpl_int_string);
  map_to_array_MG<QudaSchwarzType>(tpl_int_string, schwarz_type, QUDA_INVALID_SCHWARZ, get_schwarz_type);

  tpl_int_int.clear();
  isFound=opt.set("Q-mg-schwarz-cycle", "The number of Schwarz cycles to apply per smoother application, usage(level,int), (default=1)", false, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, schwarz_cycle, 1, 0, 127, "ERROR: invalid Schwarz cycle value requested");

  tpl_int_double.clear();
  isFound=opt.set("Q-mg-smoother-tol", "The smoother tolerance to use for each multigrid, usage(level,float(0,1)), (default 0.25)", false, tpl_int_double);
  map_to_array_MG<double>(tpl_int_double, smoother_tol, 0.25, 0, 1, "ERROR: invalid tolerance for the MG smoother");

  tpl_int_int.clear();
  isFound=opt.set("Q-mg-setup-iters", "The number of setup iterations to use for the multigrid, usage (level,int), (default 1)", false, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, num_setup_iter, 1, 0, 10, "ERROR: invalid max number of MG setup iterations refresh");
  
  opt.set("Q-mg-pre-orth", "If orthonormalize the vector before inverting in the setup of multigrid (default false)", false, pre_orthonormalize);
  opt.set("Q-mg-post-orth", "If orthonormalize the vector after inverting in the setup of multigrid (default false)", false, post_orthonormalize);

  if(showThem && !opt.getIsHelp()) printBasicOptionsWsolver();
}

  //  isFound=opt.set("Q-prec-refine", "Sloppy precision for refinement in the GPU, options (double,single,half),default (invalid)", visualize, tmpString);
  //if(isFound)prec_refinement_sloppy = get_prec(tmpString.c_str());

  //  tpl_int_string.clear();
  // isFound=opt.set("Q-mg-coarse-solve-type", "The type of solve to do on each level, usage(level,solve), (direct, direct-pc) (default = solve_type)", visualize, tpl_int_string);
  // if(isFound) map_to_array_MG<QudaSolveType>(tpl_int_string, coarse_solve_type, get_solve_type);

  // tpl_int_string.clear();
  // isFound=opt.set("Q-mg-smoother-solve-type", "The type of solve to do on smoother, usage(level,solve), (direct, direct-pc) (default = direct-pc)", visualize, tpl_int_string);
  // if(isFound) map_to_array_MG<QudaSolveType>(tpl_int_string, smoother_solve_type, get_solve_type);

  // tpl_int_int.clear();
  // isFound=opt.set("Q-mg-setup-maxiter", "The maximum number of solver iterations to use when relaxing on a null space vector, usage (level,int), (default 500)", visualize, tpl_int_int);
  // if(isFound) map_to_array_MG<int>(tpl_int_int, setup_maxiter, 0, 10000, "ERROR: invalid max number of MG setup iterations");

  // tpl_int_int.clear();
  // isFound=opt.set("Q-mg-setup-maxiter-refresh", "The maximum number of solver iterations to use when refreshing the pre-existing null space vectors, usage(level,int), (default 100)", visualize, tpl_int_int);
  // if(isFound) map_to_array_MG<int>(tpl_int_int, setup_maxiter_refresh, 0, 10000, "ERROR: invalid max number of MG setup iterations refresh");

  //  isFound=opt.set("Q-mg-smoother-halo-prec", "The smoother halo precision (applies to all levels - defaults to null_precision)", visualize, tmpString );
  //  if(isFound) smoother_halo_prec = get_prec(tmpString.c_str());

  
