#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initialize(argc, argv);

  // This shouold create the file test.h5
  HDF5 test("./test");

  // This shouold open the file test.h5 and create group foo1
  HDF5 test2("./test.h5/foo1");

  if (latfile != "") {
    // Allocation done on BOTH, DEVICE and HOST
    PLEGMA_Gauge<double> gauge(BOTH);
    
    // Reading from Lime file and loading to device
    gauge.readFromLime(latfile.c_str());
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
    
    test.write_dataset("gauge",gauge.H_elem(), shape, lshape, start);
    test.write_attribute("gauge", "shape", descr);
    test.write_attribute("gauge", "latfile", latfile);
    test.write_attribute("gauge", "plaquette", std::to_string(plaq));
  }
  
  finalize();
 
  return 0;
}
