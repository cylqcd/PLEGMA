#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
extern char latfile[];

int main(int argc, char **argv)
{
  PLEGMA_params params;
  initialize(argc, argv, &params);

  PLEGMA_Field<double> f(BOTH,FIELD);
  f.setUnit((std::vector<int>) {0});

  PLEGMA_Propagator3D<double> prop3D;
  prop3D.setUnit((std::vector<int>) {0,1,2,3,4,5,6,7,8});
  
  PLEGMA_FT<double> ft4(0,4,true);
  ft4.apply(f);
  ft4.apply(f); // apply twice to check accumulation
  ft4.writeToDisk("/onyx/noether/h/khadjiyiannakou/runs/momTest_field_ft4.dat", ASCII_FORM);

  PLEGMA_FT<double> ft3(0,3);
  ft3.apply(prop3D);
  ft3.writeToDisk("/onyx/noether/h/khadjiyiannakou/runs/momTest_prop3D_ft3.dat", ASCII_FORM);
  
  finalize();

  return 0;
}
