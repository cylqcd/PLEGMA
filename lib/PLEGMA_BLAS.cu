#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_Thrust.h>

namespace plegma{

  template<typename Float>
  struct ElemWiseMul{
    ElemWiseMul(){}
    template<typename Tuple>
    inline __device__ void operator()(Tuple t){
      Float2<Float> &x = thrust::get<0>(t);
      Float2<Float> &y = thrust::get<1>(t);
      y = y*x;
    }
  };

  /**
     @brief Element wise multiplication of two complex vectors
     @param int NN, number of elements for each vector
     @param const Float* x, Device pointer to vector which is only input
     @param Float* y, y = y*x, Device pointer to y is input and output
     @return void
  **/
  template<typename Float>
  void elemWiseMul(int NN, Float* x, Float* y){
    Float2<Float>* x_2 = (Float2<Float>*) x;
    Float2<Float>* y_2 = (Float2<Float>*) y;
    try{
      typedef thrust::device_ptr<Float2<Float> > DevF2;
      const DevF2 dev_x_2(x_2);
      DevF2 dev_y_2(y_2);
      typedef thrust::tuple<DevF2,DevF2> tplDevF2DevF2;
      typedef thrust::zip_iterator<tplDevF2DevF2> zipTplDevF2DevF2;
      zipTplDevF2DevF2 z1 = thrust::make_zip_iterator(thrust::make_tuple(x_2,y_2));
      zipTplDevF2DevF2 z2 = thrust::make_zip_iterator(thrust::make_tuple(x_2+NN,y_2+NN));
      thrust::for_each(z1,z2,ElemWiseMul<Float>());
    }
    catch(thrust::system_error &err){
      errorQuda("Thrust error detected: %s", err.what());
    }
  }
  template void elemWiseMul<float>(int NN, float* x, float* y);
  template void elemWiseMul<double>(int NN, double* x, double* y);
}
