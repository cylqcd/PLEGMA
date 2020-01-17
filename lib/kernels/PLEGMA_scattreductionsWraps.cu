#include <PLEGMA_scattreductionsV2.cu>
#include <PLEGMA_scattreductionsV3.cu>
#include <PLEGMA_scattreductionsV4.cu>

using namespace plegma;

template<VRED V, unsigned int NG, typename FloatOut, typename FloatV, typename FloatP, typename ...Args>
void V_kernels( ProfileStruct &ps, Float2<FloatOut> *block2,
		int it, int time_step, int3 source, tex_mom_list moms,
		KernelArr<GAMMAS> &listGammas, FloatV *Phi, FloatP* S1, FloatP* S2=NULL){
  dim3 grid = ps.tp.grid;
  grid.x = (grid.x/time_step)*MIN(HGC_localL[3]-it, time_step);
  
  if((V==V_2) && (S2!=NULL))
    V2_kernel<FloatOut,FloatV,FloatP,NG><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms);
  else if((V==V_3) && (S2==NULL))
    V3_kernel<FloatOut,FloatV,FloatP,NG><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms);
  else if((V==V_4) && (S2!=NULL))
    V4_kernel<FloatOut,FloatV,FloatP,NG><<<grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, MIN(HGC_localL[3]-it, time_step), source, moms);
  else
    PLEGMA_error("Unrecognized V reduction type\n");
}

template<VRED V, typename FloatOut, typename FloatV, typename FloatP, typename ...Args>
void V_kernels_wrapper( ProfileStruct &ps, Float2<FloatOut> *block2,
			int it, int time_step, int3 source, tex_mom_list moms,
			KernelArr<GAMMAS> &listGammas, FloatV *Phi, FloatP *S1, FloatP *S2=NULL){

  switch(listGammas.size){
  case(1): V_kernels<V,(unsigned int)1,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  case(2): V_kernels<V,(unsigned int)2,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  case(3): V_kernels<V,(unsigned int)3,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  case(4): V_kernels<V,(unsigned int)4,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(5): V_kernels<V,(unsigned int)5,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(6): V_kernels<V,(unsigned int)6,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(7): V_kernels<V,(unsigned int)7,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(8): V_kernels<V,(unsigned int)8,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(9): V_kernels<V,(unsigned int)9,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(10): V_kernels<V,(unsigned int)10,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(11): V_kernels<V,(unsigned int)11,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(12): V_kernels<V,(unsigned int)12,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(13): V_kernels<V,(unsigned int)13,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(14): V_kernels<V,(unsigned int)14,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(15): V_kernels<V,(unsigned int)15,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(16): V_kernels<V,(unsigned int)16,FloatOut,FloatV,FloatP, Args...>( ps, block2, it, time_step, source, moms, listGammas, Phi, S1, S2 ); break;
  }
}
