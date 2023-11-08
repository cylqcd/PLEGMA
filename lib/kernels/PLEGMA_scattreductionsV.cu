#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_ScattCorrelator.h>
#include <PLEGMA_scattreductions.cuh>
#include <PLEGMA_scattreductionsV2.cuh>
#include <PLEGMA_scattreductionsV3.cuh>
#include <PLEGMA_scattreductionsV4.cuh>

using namespace plegma;

template<unsigned int NG, typename FloatOut, typename FloatV, typename FloatP>
void V_kernels( ProfileStruct &ps, VRED V, Float2<FloatOut> *block2,
		int it, int time_step, int maxT, int4 source, tex_mom_list moms,
		KernelArr<GAMMAS_SCATT> &listGammas,
		vectorTex<FloatV> &Phi, propTex<FloatP>& S1, propTex<FloatP>& S2){
  if(V==V_2)
    V2_kernel<FloatOut,FloatV,FloatP,NG><<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, time_step, maxT, source, moms);
  else if(V==V_3)
    V3_kernel<FloatOut,FloatV,FloatP,NG><<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, block2, it, time_step, maxT, source, moms);
  else if(V==V_4)
    V4_kernel<FloatOut,FloatV,FloatP,NG><<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>(Phi, listGammas, S1, S2, block2, it, time_step, maxT, source, moms);
  else
    PLEGMA_error("Unrecognized V reduction type\n");
}

template<typename FloatOut, typename FloatV, typename FloatP>
void V_kernels_wrapper( ProfileStruct &ps, VRED V, Float2<FloatOut> *block2,
			int it, int time_step, int maxT, int4 source, tex_mom_list moms,
			KernelArr<GAMMAS_SCATT> &listGammas,
			vectorTex<FloatV> &Phi, propTex<FloatP>& S1, propTex<FloatP>& S2){

  switch(listGammas.size){
    //case(1): V_kernels<(unsigned int)1,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
    //case(4): V_kernels<(unsigned int)4,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
    //case(9): V_kernels<(unsigned int)9,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  //case(4): V_kernels<(unsigned int)4,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(5): V_kernels<(unsigned int)5,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(6): V_kernels<(unsigned int)6,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(7): V_kernels<(unsigned int)7,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(8): V_kernels<(unsigned int)8,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(9): V_kernels<(unsigned int)9,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(10): V_kernels<(unsigned int)10,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(11): V_kernels<(unsigned int)11,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(12): V_kernels<(unsigned int)12,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(13): V_kernels<(unsigned int)13,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(14): V_kernels<(unsigned int)14,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(15): V_kernels<(unsigned int)15,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  // case(16): V_kernels<(unsigned int)16,FloatOut,FloatV,FloatP>( ps, V, block2, it, time_step, maxT, source, moms, listGammas, Phi, S1, S2 ); break;
  default: PLEGMA_error("not initialized for nGammas=%d\n",listGammas.size);
  }
}


template
void V_kernels_wrapper<float, float, float>( ProfileStruct &, VRED, Float2<float> *,
					     int, int, int, int4, tex_mom_list,
					     KernelArr<GAMMAS_SCATT> &,
					     vectorTex<float>&, propTex<float>&, propTex<float>&);

template
void V_kernels_wrapper<double, double, double>( ProfileStruct &, VRED, Float2<double> *,
						int, int, int, int4, tex_mom_list,
						KernelArr<GAMMAS_SCATT> &,
						vectorTex<double>&, propTex<double>&, propTex<double>&);
