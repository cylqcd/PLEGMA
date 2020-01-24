#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

static std::vector<std::string> listOpt = {"verbosity", "load-gauge"};

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt); // Put list of options later
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  int n_GFsteps = 10, n_EachMeas = 1;
  double epsilon = 0.01;
  std::string outFile;
  std::string traj_id;
  
  HGC_options->set("GF_epsilon", "Size of discretization step used during the integration of Gradient Flow equation", verbosity, epsilon);
  HGC_options->set("n_GFsteps", "Total number of gradient flow steps", verbosity, n_GFsteps);
  HGC_options->set("n_EachMeas", "Number of steps between two measurements of Q", verbosity, n_EachMeas);
  HGC_options->set("output", "Path of desired output file", verbosity, outFile);
  HGC_options->set("traj_id", "Trajectory id number", verbosity, traj_id);

  if( outFile.empty() ) PLEGMA_error("Insert --output option with name of output file");
  if( traj_id.empty() ) PLEGMA_error("Insert --traj_id option with the number of trajectory");
  //=========================================================================================================//
  initializePLEGMA();

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> gauge(BOTH);
  PLEGMA_Gauge<double> Z_aux(DEVICE);
  double plaq, q_plaq, q_clov;
  int n_cicle = int( n_GFsteps/n_EachMeas);

  FILE *ptr_out = NULL;
  int rank;

  MPI_Initialized( &rank );
  if( rank ) MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  else rank=0;
  
  if(rank == 0){
    ptr_out = fopen(outFile.c_str(), "a");
    if(ptr_out == NULL) PLEGMA_error("Error opening file for writing\n");
  }
  
  Z_aux.zero_where(DEVICE);

  // Reading from Lime file and loading to device
  gauge.readFile( latfile, LIME_FORMAT );
  gauge.load();
  
  // Compuiting plaquette on device in three different way for crosschecking
  plaq = gauge.calculatePlaq( );
  
  PLEGMA_printf("* flowstep: 0.00 *\n");
  q_plaq = gauge.calculateTopo( PLAQUETTE );
  q_clov = gauge.calculateTopo( CLOVER );
  PLEGMA_printf("****************\n"); 
  if(rank == 0){
    fprintf( ptr_out, "%s\t%g\t%.14f\t%.14f\t%.14f\n", traj_id.c_str(), 0., q_plaq, q_clov, plaq );
  }
  
  for(int i=0; i<n_cicle; i++){
    PLEGMA_printf("* flowstep: %g *\n", (i+1)*n_EachMeas*epsilon);
    gauge.applyGradientFlow( Z_aux, n_EachMeas, epsilon );
    q_plaq = gauge.calculateTopo( PLAQUETTE );
    q_clov = gauge.calculateTopo( CLOVER );
    plaq = gauge.calculatePlaq();
    PLEGMA_printf("****************\n");
    if(rank == 0){
      fprintf( ptr_out, "%s\t%g\t%.14f\t%.14f\t%.14f\n", traj_id.c_str(), (i+1)*n_EachMeas*epsilon, q_plaq, q_clov, plaq );
    }
  }
    
  // Loading to QUDA and computing plaquette also there
  if(rank == 0){
    fclose( ptr_out );
  }
  
  finalize();
 
  return 0;
}
