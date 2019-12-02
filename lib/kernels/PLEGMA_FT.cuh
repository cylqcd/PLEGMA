#include <PLEGMA_Field.h>
#include <PLEGMA_FT.h>
#include <PLEGMA_kernel_utils.cuh>

using namespace plegma;
template<typename Float>
struct MomF{
  int sign;
  Float momx, momy, momz, momt;
  MomF(int sign, Float momx, Float momy, Float momz, Float momt):sign(sign),momx(momx),momy(momy),momz(momz),momt(momt){}
  template<typename Tuple>
  __device__ void operator()(Tuple t){
    int id = thrust::get<0>(t);
    int x[4] = GET_ID(id);
#pragma unroll
    for(int i=0; i<4; i++)
      x[i] += DGC_procPosition[i] * DGC_localL[i];
    Float phase = ((Float) momx*x[0]) / ((Float) DGC_totalL[0])  +
      ((Float) momy*x[1]) / ((Float) DGC_totalL[1]) +
      ((Float) momz*x[2]) / ((Float) DGC_totalL[2]) +
      ((Float) momt*x[3]) / ((Float) DGC_totalL[3]);
    Float2<Float> &el = (thrust::get<1>(t));
    phase *= 2. * PI;
    el.x = cos(phase);
    el.y = (sign == +1) ? sin(phase): -sin(phase);
  }
};

template<typename Float>
static void createMomField(Float2<Float> *x, std::vector<Float> mom, int D3D4, int sign){
  if(D3D4 == 3){
    if(mom.size() != 3) PLEGMA_error("A momentum vector in three dimensions need three components\n");}
  else if (D3D4 == 4){
    if(mom.size() != 4) PLEGMA_error("A momentum vector in four dimensions need four components\n");}
  else PLEGMA_error("Not supported");
  int V = (D3D4 == 3) ? HGC_localVolume/HGC_localL[3] : HGC_localVolume;
  thrust::counting_iterator<int> first(0);
  thrust::counting_iterator<int> last = first + V;
  thrust::device_ptr<Float2<Float> > dev_ptr(x);
  typedef thrust::tuple<thrust::counting_iterator<int>, thrust::device_ptr<Float2<Float> > > tplIntDev;
  typedef thrust::zip_iterator<tplIntDev> zipTplIntDev;
  zipTplIntDev z1 = thrust::make_zip_iterator(thrust::make_tuple(first,dev_ptr));
  zipTplIntDev z2 = thrust::make_zip_iterator(thrust::make_tuple(last,dev_ptr + V));
  if(D3D4 == 3) thrust::for_each(z1,z2,MomF<Float>(sign,mom[0],mom[1],mom[2],0));
  else thrust::for_each(z1,z2,MomF<Float>(sign,mom[0],mom[1],mom[2],mom[3]));
}


template<typename Float>
static void FT_dot(PLEGMA_FT<Float> &ft, const PLEGMA_Field<Float> &f, std::vector<std::vector<Float> > mom, int sign){
  if(sign != +1 && sign != -1) PLEGMA_error("Sign should be either +1 or -1\n");
  if(mom.size() == 0) PLEGMA_error("Momentum container is empty");
  int Nmom = mom.size();
  int V3 = HGC_localVolume/HGC_localL[3];
  int V = ft.Dims() == 3 ? V3 : HGC_localVolume;
  Float2<Float> *x;
  cudaMalloc((void**)&x, V*2*sizeof(Float));
  cudaMemset(x,0,V*2*sizeof(Float));
  checkCudaError();
  for(int imom = 0; imom < Nmom; imom++){
    createMomField(x,mom[imom],ft.Dims(),-sign); // change sign to compensate dagger
    for(int idf = 0 ; idf < f.Field_length(); idf++)
      for(int it = 0 ; it < ft.DimT(); it++){
	Float2<Float> *y = (Float2<Float> *)f.D_elem() + idf*f.Total_length() + it*V3;
	std::complex<Float> res = cuBLAS::dot((ft.Dims() == 3) ? V3 : HGC_localVolume, (Float*) x,(Float*) y,
					      (ft.Dims() == 3) ? HGC_spaceComm : HGC_fullComm);
	ft.H_elem()[it*f.Field_length()*Nmom*2 + idf*Nmom*2 + imom*2 + 0] += res.real();
	ft.H_elem()[it*f.Field_length()*Nmom*2 + idf*Nmom*2 + imom*2 + 1] += res.imag();
      }
  }
  cudaFree(x);
}

template<typename Float>
static void FT_gemv(PLEGMA_FT<Float> &ft, const PLEGMA_Field<Float> &f, std::vector<std::vector<Float> > mom, int sign){
  if(sign != +1 && sign != -1) PLEGMA_error("Sign should be either +1 or -1\n");
  if(mom.size() == 0) PLEGMA_error("Momentum container is empty");
  int Nmom = mom.size();
  int V3 = HGC_localVolume/HGC_localL[3];
  int V = ft.Dims() == 3 ? V3 : HGC_localVolume;
  Float2<Float> *x,*d_res;
  cudaMalloc((void**)&x, V*2*sizeof(Float));
  cudaMemset(x,0,V*2*sizeof(Float));
  cudaMalloc((void**)&d_res, f.Field_length() * ft.DimT() * 2*sizeof(Float));
  checkCudaError();
  Float2<Float> h_res[f.Field_length()*ft.DimT()];
  Float2<Float> *h_ft = (Float2<Float> *) ft.H_elem();
  Float one[2] = {1.,0.}, zero[2] = {0.,0.};
  for(int imom = 0; imom < Nmom; imom++){
    createMomField(x,mom[imom],ft.Dims(),sign); 
    cuBLAS::gemv(TRANS,(ft.Dims() == 3) ? V3 : HGC_localVolume, f.Field_length() * ft.DimT(), one,
		 (Float*) f.D_elem(), (Float*) x, zero, (Float*) d_res, (Float*) h_res,
		 (ft.Dims() == 3) ? HGC_spaceComm : HGC_fullComm);
    for(int idf = 0 ; idf < f.Field_length(); idf++)
      for(int it = 0 ; it < ft.DimT(); it++)
	  h_ft[it*f.Field_length()*Nmom + idf*Nmom + imom] += h_res[idf*ft.DimT()+it];
  }
  cudaFree(x);
  cudaFree(d_res);
}



template<typename Float>
__global__ void fourier_transform_3D_kernel(Float* block, pFloat2<Float> in, tex_mom_list texMomList,int it, int sign){
  int vid = blockIdx.x*blockDim.x + threadIdx.x;
  int sid = vid + it*DGC_localVolume3D;
  Float2<Float> *block2 = (Float2<Float> *)block;
  if(vid >= DGC_localVolume3D) return;
  Float2<Float> tmp;
  extern __shared__ int ext_shared_cache[];
  Float2<Float> *shared_cache = (Float2<Float> *) ext_shared_cache;
  int source_pos[3] = {0,0,0};      

  in.setSid(sid);
  for(int i = 0; i < in.site_size; i++){
    tmp = in.get(i);  
    if(block2==NULL) fourier_transform_3D(block2,&tmp,shared_cache,1,vid,source_pos,texMomList,in.site_size-1,sign);
    else fourier_transform_3D(block2+i*gridDim.x,&tmp,shared_cache,1,vid,source_pos,texMomList,in.site_size-1,sign);
  }
}


template<typename Float>
static void fourier_transform_3D_k(PLEGMA_FT<Float> &ft, const PLEGMA_Field<Float> &field, tex_mom_list &texMomList, int it, int sign){
  int SpVol = HGC_localVolume/HGC_localL[3];
  Float *d_partial_block = NULL;
  int site_size = field.Field_length();
  int shared_size = 2*sizeof(Float);
  ProfileStruct ps(SpVol, shared_size);
  tune(ps, "fourier_transform_3D_kernel", fourier_transform_3D_kernel<Float>, d_partial_block, toField2<pFloat2>(field), texMomList, it, sign);
  size_t alloc_size=ft.Nmoms()*site_size*ps.tp.grid.x*2*sizeof(Float);
  cudaMalloc((void**)&d_partial_block, alloc_size);
  checkCudaError();
  run(ps, "fourier_transform_3D_kernel", fourier_transform_3D_kernel<Float>, d_partial_block, toField2<pFloat2>(field), texMomList, it, sign);
  Float *h_partial_block = NULL;
  hostMalloc(h_partial_block,alloc_size);
  cudaMemcpy(h_partial_block,d_partial_block,alloc_size,cudaMemcpyDeviceToHost);
  checkCudaError();
  cudaFree(d_partial_block);
  int gridDimX = ps.tp.grid.x;
  Float *reduction;
  hostMalloc(reduction,ft.Nmoms()*site_size*2*sizeof(Float));
  for(int i = 0 ; i < ft.Nmoms()*site_size; i++){
    reduction[i*2+0] = 0;
    reduction[i*2+1] = 0;
    for(int j = 0 ; j < gridDimX; j++) {
      reduction[i*2+0] += h_partial_block[(i*gridDimX + j)*2+0];
      reduction[i*2+1] += h_partial_block[(i*gridDimX + j)*2+1];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,reduction, ft.Nmoms()*site_size*2, MPI_Type(reduction), MPI_SUM, HGC_spaceComm);
  for(int imom = 0; imom < ft.Nmoms(); imom++)
    for(int idf = 0 ; idf < field.Field_length(); idf++)
      for(int ir = 0 ; ir < 2 ; ir++)
	ft.H_elem()[it*field.Field_length()*ft.Nmoms()*2 + idf*ft.Nmoms()*2 + imom*2 + ir] += reduction[imom*field.Field_length()*2+idf*2+ir];
  hostFree(reduction,ft.Nmoms()*site_size*2*sizeof(Float));
  hostFree(h_partial_block,alloc_size);
}
