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
  std::vector<std::vector<int> > mom;
  mom.push_back((std::vector<int>) {0,0,0});
  mom.push_back((std::vector<int>) {1,1,1});
  double *resMom = f.FT3D(mom);
  for(int i = 0 ; i < GK_localL[3]; i++)
    for(int imom = 0 ; imom < mom.size(); imom++)
      printfQuda("%+d %+d %+d %+d %f %f\n", i,mom[imom][0],mom[imom][1],mom[imom][2], resMom[i*mom.size()*2 + imom*2], resMom[i*mom.size()*2 + imom*2 + 1]);
  free(resMom);
  finalize();

  return 0;
}
