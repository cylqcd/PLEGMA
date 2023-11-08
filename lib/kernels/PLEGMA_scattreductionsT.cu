#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_ScattCorrelator.h>
#include <PLEGMA_scattreductions.cuh>
#include <PLEGMA_scattreductionsT1.cuh>
#include <PLEGMA_scattreductionsT2.cuh>

using namespace plegma;

template<unsigned int NGI, unsigned int NGF, typename FloatOut, typename FloatP>
void T_kernels( ProfileStruct &ps, TRED T, Float2<FloatOut> *block2,
		int it, int time_step, int maxT, int4 source, tex_mom_list moms,
		KernelArr<GAMMAS_SCATT> &listGammas_i, KernelArr<GAMMAS_SCATT> &listGammas_f,
		propTex<FloatP>& S1, propTex<FloatP>& S2, propTex<FloatP>& S3){
  
  if( T==T_1 )
    T1_kernel<FloatOut,FloatP,NGI,NGF><<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>(listGammas_i,listGammas_f, S1, S2, S3, block2, it, time_step, maxT, source, moms);
  else if( T==T_2 )
    T2_kernel<FloatOut,FloatP,NGI,NGF><<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>(listGammas_i,listGammas_f, S1, S2, S3, block2, it, time_step, maxT, source, moms);
  else
    PLEGMA_error("Unrecognized V reduction type\n");
}


template<typename FloatOut, typename FloatP>
void T_kernels_wrapper( ProfileStruct &ps, TRED T, Float2<FloatOut> *block2,
			int it, int time_step, int maxT, int4 source, tex_mom_list moms,
			KernelArr<GAMMAS_SCATT> &listGammas_i, KernelArr<GAMMAS_SCATT> &listGammas_f,
			propTex<FloatP>& S1, propTex<FloatP>& S2, propTex<FloatP>& S3 ){

  switch(listGammas_i.size){
  case(6):
    switch(listGammas_f.size){
    case(6): T_kernels<(unsigned int)6,(unsigned int)6,FloatOut,FloatP>( ps, T, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
    default: PLEGMA_error("not initialized for nGammas_f=%d\n",listGammas_f.size);
    }; break;
  default: PLEGMA_error("not initialized for nGammas_i=%d\n",listGammas_i.size);
  }
}


template
void T_kernels_wrapper<float, float> ( ProfileStruct&, TRED, Float2<float>*,
				       int, int, int, int4, tex_mom_list,
				       KernelArr<GAMMAS_SCATT>&, KernelArr<GAMMAS_SCATT>&,
				       propTex<float>&, propTex<float>&, propTex<float>&);

template
void T_kernels_wrapper<double, double> ( ProfileStruct&, TRED, Float2<double>*,
				       int, int, int, int4, tex_mom_list,
				       KernelArr<GAMMAS_SCATT>&, KernelArr<GAMMAS_SCATT>&,
				       propTex<double>&, propTex<double>&, propTex<double>&);
