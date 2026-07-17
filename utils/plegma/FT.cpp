#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, false); // Add list of options
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  //=========================================================================================================//
  initializePLEGMA();


  PLEGMA_Field<double> f(BOTH,SCALAR);
  f.setUnit((std::vector<int>) {0});

  PLEGMA_Propagator3D<double> prop3D;
  prop3D.setUnit((std::vector<int>) {0,1,2,3,4,5,6,7,8});
  
  PLEGMA_FT<double> ft4(1,4,true);
  ft4.apply(f);
  ft4.apply(f); // apply twice to check accumulation
  ft4.writeFile("./momTest_field_ft4.dat", ASCII_FORMAT);

  f.mulMomentumPhases((std::vector<int>) {+1,0,0,0});
  
  PLEGMA_FT<double> ft3(1,3);
  ft3.apply(prop3D);
  ft3.mulConstMomentumPhases((std::vector<int>) {1,2,3}, +1 );
  ft3.writeFile("./momTest_prop3D_ft3.dat", ASCII_FORMAT);
  
  finalize();

  return 0;
}
