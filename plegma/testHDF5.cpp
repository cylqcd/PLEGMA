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
    PLEGMA_FT<double> ft3(1,3);
    PLEGMA_FT<double> ft4(1,4);
    PLEGMA_Field<double> f(BOTH,SCALAR);
    PLEGMA_Propagator3D<double> prop3D;

    f.setUnit((std::vector<int>) {0});
    prop3D.setUnit((std::vector<int>) {0,1,2,3,4,5,6,7,8});
     
    ft3.apply(f, FT_GEMV);
    ft3.writeFile("./momTest.h5/scalar/3D/gemv", HDF5_FORMAT);
    ft3.zero();
    ft3.apply(f, FT_NAIVE);
    ft3.writeFile("./momTest.h5/scalar/3D/naive", HDF5_FORMAT);
    ft3.zero();
    ft4.apply(f, FT_GEMV);
    ft4.writeFile("./momTest.h5/scalar/4D/gemv", HDF5_FORMAT);
    ft4.zero();

    ft3.apply(prop3D, FT_GEMV);
    ft3.writeFile("./momTest.h5/prop3D/3D/gemv", HDF5_FORMAT);
    ft3.zero();
    ft3.apply(prop3D, FT_NAIVE);
    ft3.writeFile("./momTest.h5/prop3D/3D/naive", HDF5_FORMAT);
    ft3.zero();
  }
  
  finalize();
 
  return 0;
}
