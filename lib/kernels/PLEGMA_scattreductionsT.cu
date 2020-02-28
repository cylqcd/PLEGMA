#include <PLEGMA_scattreductionsT1.cuh>
#include <PLEGMA_scattreductionsT2.cuh>

using namespace plegma;

template<TRED T, unsigned int NGI, unsigned int NGF, typename FloatOut, typename FloatP>
void T_kernels( ProfileStruct &ps, Float2<FloatOut> *block2,
		int it, int time_step, int maxT, int4 source, tex_mom_list moms,
		KernelArr<GAMMAS_SCATT> &listGammas_i, KernelArr<GAMMAS_SCATT> &listGammas_f,  FloatP* S1, FloatP* S2, FloatP* S3){
  
  if( T==T_1 )
    T1_kernel<FloatOut,FloatP,NGI,NGF><<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>(listGammas_i,listGammas_f, S1, S2, S3, block2, it, time_step, maxT, source, moms);
  else if( T==T_2 )
    T2_kernel<FloatOut,FloatP,NGI,NGF><<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>(listGammas_i,listGammas_f, S1, S2, S3, block2, it, time_step, maxT, source, moms);
  else
    PLEGMA_error("Unrecognized V reduction type\n");
}


template<TRED T,typename FloatOut, typename FloatP>
void T_kernels_wrapper( ProfileStruct &ps, Float2<FloatOut> *block2,
			int it, int time_step, int maxT, int4 source, tex_mom_list moms,
			KernelArr<GAMMAS_SCATT> &listGammas_i, KernelArr<GAMMAS_SCATT> &listGammas_f,
			FloatP *S1, FloatP *S2, FloatP *S3 ){

  switch(listGammas_i.size){
  case(1):
    switch(listGammas_f.size){
    case(1): T_kernels<T,(unsigned int)1,(unsigned int)1,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
    case(2): T_kernels<T,(unsigned int)1,(unsigned int)2,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
    // case(3): T_kernels<T,(unsigned int)1,(unsigned int)3,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
    // case(4): T_kernels<T,(unsigned int)1,(unsigned int)4,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
    default: PLEGMA_error("not initialized for nGammas_f=%d\n",listGammas_f.size);
    }; break;
  case(2):
    switch(listGammas_f.size){
    case(1): T_kernels<T,(unsigned int)2,(unsigned int)1,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
    case(2): T_kernels<T,(unsigned int)2,(unsigned int)2,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
    // case(3): T_kernels<T,(unsigned int)2,(unsigned int)3,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
    // case(4): T_kernels<T,(unsigned int)2,(unsigned int)4,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
    default: PLEGMA_error("not initialized for nGammas_f=%d\n",listGammas_f.size);
    }; break;
  // case(3):
  //   switch(listGammas_f.size){
  //   case(1): T_kernels<T,(unsigned int)3,(unsigned int)1,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
  //   case(2): T_kernels<T,(unsigned int)3,(unsigned int)2,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
  //   case(3): T_kernels<T,(unsigned int)3,(unsigned int)3,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
  //   case(4): T_kernels<T,(unsigned int)3,(unsigned int)4,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
  //   default: PLEGMA_error("not initialized for nGammas_f=%d\n",listGammas_f.size);
  //   }; break;
  // case(4):
  //   switch(listGammas_f.size){
  //   case(1): T_kernels<T,(unsigned int)4,(unsigned int)1,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
  //   case(2): T_kernels<T,(unsigned int)4,(unsigned int)2,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
  //   case(3): T_kernels<T,(unsigned int)4,(unsigned int)3,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
  //   case(4): T_kernels<T,(unsigned int)4,(unsigned int)4,FloatOut,FloatP>( ps, block2, it, time_step, maxT, source, moms, listGammas_i, listGammas_f, S1, S2, S3); break;
  //   default: PLEGMA_error("not initialized for nGammas_f=%d\n",listGammas_f.size);
  //   }; break;
    
  default: PLEGMA_error("not initialized for nGammas_i=%d\n",listGammas_i.size);
  }
}


