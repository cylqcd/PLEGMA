#include <PLEGMA_Field.h>
#include <PLEGMA_FT.h>
#include <PLEGMA_kernel_utils.cuh>

using namespace plegma;
template<typename Float>
struct MomF{
  int sign;
  int momx, momy, momz, momt;
  MomF(int sign, int momx, int momy, int momz, int momt):sign(sign),momx(momx),momy(momy),momz(momz),momt(momt){}
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
static void createMomField(Float2<Float> *x, std::vector<int> mom, int D3D4, int sign){
  if(D3D4 == 3){
    if(mom.size() != 3) errorQuda("A momentum vector in three dimensions need three components\n");}
  else if (D3D4 == 4){
    if(mom.size() != 4) errorQuda("A momentum vector in four dimensions need four components\n");}
  else errorQuda("Not supported");
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
static void FT(PLEGMA_FT<Float> &ft, const PLEGMA_Field<Float> &f, std::vector<std::vector<int> > mom, int sign){
  if(sign != +1 && sign != -1) errorQuda("Sign should be either +1 or -1\n");
  if(mom.size() == 0) errorQuda("Momentum container is empty");
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
	std::complex<Float> res, lres = cuBLAS::dot((ft.Dims() == 3) ? V3 : HGC_localVolume, (Float*) x,(Float*) y);
	int mpiErr = MPI_Allreduce((Float*) &lres, (Float*) &res, 2,
				   MPI_Type<Float>(), MPI_SUM, (ft.Dims() == 3) ?
				   HGC_spaceComm : MPI_COMM_WORLD);
	if(mpiErr != MPI_SUCCESS)
	  errorQuda("MPI_Allreduce failed with error %d\n", mpiErr);

	ft.H_elem()[it*f.Field_length()*Nmom*2 + idf*Nmom*2 + imom*2 + 0] += res.real();
	ft.H_elem()[it*f.Field_length()*Nmom*2 + idf*Nmom*2 + imom*2 + 1] += res.imag();
      }
  }
  cudaFree(x);
}
