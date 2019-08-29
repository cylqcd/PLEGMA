#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, false); // Add list of options
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  //=========================================================================================================//
  initializePLEGMA();

  { 
    // This creates the file foo.h5
    // with directory structure /foo1/foo2/foo3/foo4/foo5
    HDF5 test("./foo.h5/foo1");
    test.cd("foo2/");

    test.cd("../foo2/foo3");

    test.cd("/foo1/foo2/foo3/foo4/");
    
    test.cd("///foo1///./../foo1/foo2/foo3///./../../foo2/foo3/foo4/foo5");
    PLEGMA_printf("Now I'm here: %s\n",test.pwd().c_str());    
  }
  if (latfile != "") {
    // This writes a read config into conf.h5
    // Allocation done on BOTH, DEVICE and HOST
    PLEGMA_Gauge<double> gauge(BOTH);
      
    // Reading from Lime file and loading to device
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.load();
      
    // Compuiting plaquette on device in three different way for crosschecking
    double plaq = gauge.calculatePlaq();
    gauge.writeFile("./conf", HDF5_FORMAT);
  }
  {
    // This does the same as FT.cpp and writes the result in HDF5 format
    PLEGMA_Field<double> f(BOTH,SCALAR);
    f.setUnit((std::vector<int>) {0});
    
    PLEGMA_Propagator3D<double> prop3D;
    prop3D.setUnit((std::vector<int>) {0,1,2,3,4,5,6,7,8});
    
    PLEGMA_FT<double> ft4(1,4,true);
    ft4.apply(f);
    ft4.apply(f); // apply twice to check accumulation
    ft4.writeFile("./momTest_field_ft4", HDF5_FORMAT);
    
    f.mulMomentumPhases((std::vector<int>) {+1,0,0,0});
    
    PLEGMA_FT<double> ft3(1,3);
    ft3.apply(prop3D);
    ft3.mulConstMomentumPhases((std::vector<int>) {1,2,3}, +1 );
    ft3.writeFile("./momTest_prop3D_ft3", HDF5_FORMAT);
  }
  
  finalize();
 
  return 0;
}
