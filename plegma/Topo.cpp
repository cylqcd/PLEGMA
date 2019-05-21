#include <PLEGMA.h>
#include <PLEGMA_utils.h>
using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initializeOptions(argc, argv); // Put list of options later
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  //=========================================================================================================//
  initializePLEGMA();

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> gauge(BOTH);
  PLEGMA_Gauge<double> Z_aux(DEVICE);
  double t_step = 0.1;
  
  Z_aux.zero_where(DEVICE);

  // Reading from Lime file and loading to device
  gauge.readFromLime( latfile.c_str() );
  gauge.load();
  
  // Compuiting plaquette on device in three different way for crosschecking
  gauge.calculatePlaq( );
  
  PLEGMA_printf("* flowstep: 0.00 *\n");
  gauge.calculateTopo( PLAQUETTE );
  gauge.calculateTopo( CLOVER );
  PLEGMA_printf("****************\n");
  
  for(int i=0; i<20; i++){
    PLEGMA_printf("* flowstep: %g *\n", (i+1)*t_step);
    gauge.applyGradientFlow( Z_aux, 1, t_step );
    gauge.calculateTopo( PLAQUETTE );
    gauge.calculateTopo( CLOVER );
    PLEGMA_printf("****************\n");
  }
    
  // Loading to QUDA and computing plaquette also there
  
  finalize();
 
  return 0;
}
