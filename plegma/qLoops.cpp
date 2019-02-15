#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <PLEGMA_Hprobing.h>
#include <stdio.h>
using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initialize(argc, argv);

#ifdef CHECK_HPROP
  PLEGMA_Hprobing hprop(3);
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<double> vectorAuxDD;
  vectorAuxD.setUnit((std::vector<int>) {0});
  FILE *ptr_test = NULL;
  std::string strM = "/onyx/noether/h/khadjiyiannakou/runs/Hhad";
  for(int ih = 0; ih < hprop.get_NHad(); ih++){
    ptr_test = fopen((strM+std::to_string(ih)).c_str(),"w");
    vectorAuxDD.applyHpropColoring4D(vectorAuxD,hprop,ih,(std::vector<int>) {0});
    vectorAuxDD.unload();
    for (int i = 0; i < vectorAuxDD.Total_length(); ++i) {
      fprintf(ptr_test,"%d %d\n",(int) vectorAuxDD.H_elem()[i*2],(int) vectorAuxDD.H_elem()[i*2+1] );
    }
    fclose(ptr_test);
  }
  exit(-1);
#endif // 
  
  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFromLime(latfile.c_str());
  gauge.load();
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true);
  plaqQuda();

  // ensuring mu negative
  if(mu>0) mu*=-1.;
  QUDA_solver *solverDN = new QUDA_solver(mu);
  PLEGMA_Vector<double> source(DEVICE);
  PLEGMA_Vector<double> phi;
  PLEGMA_Vector<double> tmp;
  bool isOneD = true;
  PLEGMA_QLoops<double> loops_std(BOTH,isOneD);
  QudaInvertParam inv_params = solverDN->getInvParams();
  // just put units to the whole for debugging
  source.setUnit((std::vector<int>) {0,1,2,3,4,5,6,7,8,9,10,11});
  solverDN->solve(phi,source);
  // for convention reasons for quark loops we put the normalization factors of the fields later in the analysis
  phi.scaleVector(1./(2.*inv_params.kappa)); 

  gauge.communicateGhost();
  loops_std.oneEnd_trick(phi,phi,tmp,gauge,-1.,true); //standard one-end trick

  std::string prefix = "/onyx/noether/h/dnole/runs/";
  PLEGMA_FT<double> ft(1, 3);

  // do the FT and write to File std trick
  loops_std.load(loops_std.H_loc());
  ft.apply(loops_std);
  ft.writeToFile(prefix + "std_local_loops_FT.0000.dat", ASCII_FORM);
  if(isOneD)
    for(int mu = 0 ; mu < 4 ; mu++){
      loops_std.load(loops_std.H_oneD()[mu]);
      ft.apply(loops_std);
      ft.scale(0.25);
      ft.writeToFile(prefix + "std_oneD_" + std::to_string(mu) + "_loops_FT.0000.dat", ASCII_FORM);

      loops_std.load(loops_std.H_oneDC()[mu]);
      ft.apply(loops_std);
      ft.scale(0.25);
      ft.writeToFile(prefix + "std_oneDC_" + std::to_string(mu) + "_loops_FT.0000.dat", ASCII_FORM);      
    }
  

  // int rank = comm_rank();
  // loops_std.write_ASCII(prefix+"std_local_loops.0000.dat" + std::to_string(rank),
  // 			prefix+"std_oneD_loops.0000.dat" + std::to_string(rank),
  // 			prefix+"std_oneDC_loops.0000.dat" + std::to_string(rank));

  PLEGMA_QLoops<double> loops_gen(BOTH,isOneD);
  PLEGMA_Vector<double> phi_r;
  QUDA_dirac *D = nullptr;
  if(inv_params.dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
    D = new QUDA_dirac(QUDA_CLOVER_WILSON_DSLASH);
  else if (inv_params.dslash_type == QUDA_TWISTED_MASS_DSLASH)
    D = new QUDA_dirac(QUDA_WILSON_DSLASH);
  else
    errorQuda("Only QUDA_TWISTED_CLOVER_DSLASH and QUDA_TWISTED_MASS_DSLASH are allowed for the one-end trick");

  D->apply<M>(phi_r,phi);
  phi_r.apply_gamma5();
  loops_gen.oneEnd_trick(phi, phi_r, tmp, gauge, +1., true); //generalized one-end trick

  // do the FT and write to File std trick
  loops_gen.load(loops_gen.H_loc());
  ft.apply(loops_gen);
  ft.writeToFile(prefix + "gen_local_loops_FT.0000.dat", ASCII_FORM);
  if(isOneD)
    for(int mu = 0 ; mu < 4 ; mu++){
      loops_gen.load(loops_gen.H_oneD()[mu]);
      ft.apply(loops_gen);
      ft.scale(0.25);
      ft.writeToFile(prefix + "gen_oneD_" + std::to_string(mu) + "_loops_FT.0000.dat", ASCII_FORM);

      loops_gen.load(loops_gen.H_oneDC()[mu]);
      ft.apply(loops_gen);
      ft.scale(0.25);
      ft.writeToFile(prefix + "gen_oneDC_" + std::to_string(mu) + "_loops_FT.0000.dat", ASCII_FORM);      
    }

  // loops_gen.write_ASCII(prefix+"gen_local_loops.0000.dat" + std::to_string(rank),
  // 			prefix+"gen_oneD_loops.0000.dat" + std::to_string(rank),
  // 			prefix+"gen_oneDC_loops.0000.dat" + std::to_string(rank));

  delete D;
  delete solverDN;
  finalize();

  return 0;
}
