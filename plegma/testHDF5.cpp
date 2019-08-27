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
    // This creates the file conf.h5
    HDF5 test2("./conf");

    // Allocation done on BOTH, DEVICE and HOST
    PLEGMA_Gauge<double> gauge(BOTH);
      
    // Reading from Lime file and loading to device
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.load();
      
    // Compuiting plaquette on device in three different way for crosschecking
    double plaq = gauge.calculatePlaq();

    std::string descr = "shape: ";
    std::vector<hsize_t> shape, lshape, start;
    descr += "/dirs";
    shape.push_back(4);
    lshape.push_back(4);
    start.push_back(0); 
    descr += "/c1";
    shape.push_back(3);
    lshape.push_back(3);
    start.push_back(0);
    descr += "/c2";
    shape.push_back(3);
    lshape.push_back(3);
    start.push_back(0);
    descr += "/x/y/z/t";
    // Volume
    for(int i=0; i<N_DIMS; i++) {
      shape.push_back(HGC_totalL[i]);
      lshape.push_back(HGC_localL[i]);
      start.push_back((HGC_procPosition[i]*HGC_localL[i]) % HGC_totalL[i]);
    }
    descr += "/re-im";
    shape.push_back(2);
    lshape.push_back(2);
    start.push_back(0);
    
    test2.write_dataset("gauge", gauge.H_elem(), shape, lshape, start);
    test2.write_attribute("gauge", "shape", descr);
    test2.write_attribute("gauge", "latfile", latfile);
    test2.write_attribute("gauge", "plaquette", std::to_string(plaq));
  }
  
  finalize();
 
  return 0;
}
