template<typename Float>
struct MomF{
  int sign;
  int momx, momy, momz;
  __device__ MomF(int sign, int momx, int momy, int momz):sign(sign),momx(momx),momy(momy),momz(momz){}
  template<typename Tuple>
  __device__ void operator()(Tuple t){
    int id = thrust::get<0>(t);
    int x[3] = GET_ID_ZYX(id);
#pragma unroll
    for(int i=0; i<3; i++)
      x[i] += c_procPosition[i] * c_localL[i];
    Float phase = ((Float) momx*x[0]) / ((Float) c_totalL[0])  +
      ((Float) momy*x[1]) / ((Float) c_totalL[1]) + ((Float) momz*x[2]) / ((Float) c_totalL[2]);
    Float2<Float> &el = (thrust::get<1>(t));
    phase *= 2. * PI;
    el.x = cos(phase);
    el.y = (sign == +1) ? sin(phase): -sin(phase);
  }
};

template<typename Float>
static void createMomField(Float2<Float> *x, std::vector<int> mom, int sign){
  if(mom.size() != 3) errorQuda("A momentum vector in three dimensions need three components\n");
  int localVolume3D = GK_localVolume/GK_localL[3];
  thrust::counting_iterator<int> first(0);
  thrust::counting_iterator<int> last = first + localVolume3D;
  thrust::device_ptr<Float2<Float> > dev_ptr(x);
  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(first,dev_ptr)),
   		   thrust::make_zip_iterator(thrust::make_tuple(last,dev_ptr + localVolume3D)),
   		   MomF<Float>(sign,mom[0],mom[1],mom[2]));
}


template<typename Float>
static Float* FT3D_k(PLEGMA_Field<Float> &f, std::vector<std::vector<int> > mom, int sign){
  if(sign != +1 && sign != -1) errorQuda("Sign should be either +1 or -1\n");
  if(mom.size() == 0) errorQuda("Momentum container is empty");
  int Nmom = mom.size();
  int localVolume3D = GK_localVolume/GK_localL[3];
  Float2<Float> *momSpace = (Float2<Float> *) malloc(f.Field_length()*GK_localL[3]*Nmom*2*sizeof(Float));
  Float2<Float> *x;
  cudaMalloc((void**)&x, localVolume3D*2*sizeof(Float));
  cudaMemset(x,0,localVolume3D*2*sizeof(Float));
  checkCudaError();
  if(momSpace == NULL) errorQuda("Cannot allocate memory for FT3D\n");
  for(int imom = 0; imom < Nmom; imom++){
    createMomField(x,mom[imom],-sign); // change sign to compensate dagger
    for(int idf = 0 ; idf < f.Field_length(); idf++)
      for(int it = 0 ; it < GK_localL[3]; it++){
	Float2<Float> res;
	Float2<Float> *y = (Float2<Float> *)f.D_elem() + idf*GK_localVolume + it*localVolume3D;
	cuBLAS::dot((Float*) &res, localVolume3D, (Float*) x,(Float*) y, GK_spaceComm);
	momSpace[idf*Nmom + it*Nmom + imom] = res;
      }
  }
  cudaFree(x);
  return (Float*)momSpace;
}
