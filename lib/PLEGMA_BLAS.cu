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
      DevF2 dev_x_2(x_2);
      DevF2 dev_y_2(y_2);
      typedef thrust::tuple<DevF2,DevF2> tplDevF2DevF2;
      typedef thrust::zip_iterator<tplDevF2DevF2> zipTplDevF2DevF2;

      zipTplDevF2DevF2 z1 = thrust::make_zip_iterator(thrust::make_tuple(dev_x_2,dev_y_2));
      zipTplDevF2DevF2 z2 = thrust::make_zip_iterator(thrust::make_tuple(dev_x_2+NN,dev_y_2+NN));
      thrust::for_each(z1,z2,ElemWiseMul<Float>());
    }
    catch(thrust::system_error &err){
      PLEGMA_error("Thrust error detected: %s", err.what());
    }
  }
  template void elemWiseMul<float>(int NN, float* x, float* y);
  template void elemWiseMul<double>(int NN, double* x, double* y);

  //--------------------------------------------------------------
  template<typename Float>
  struct Axpbypcz{
    Float2<Float> a2;
    Float2<Float> b2;
    Float2<Float> c2;
    inline  Axpbypcz(Float2<Float> a2,Float2<Float> b2,Float2<Float> c2):a2(a2),b2(b2),c2(c2){}
    template<typename Tuple>
    inline __device__ void operator()(Tuple t){
      Float2<Float> &x = thrust::get<0>(t);
      Float2<Float> &y = thrust::get<1>(t);
      Float2<Float> &z = thrust::get<2>(t);
      z = a2*x + b2*y + c2*z;
    }
  };

  template<typename Float>
  void axpbypcz(int NN, Float a[2], Float* x, Float b[2], Float* y, Float c[2], Float* z){
    Float2<Float>* x_2 = (Float2<Float>*) x;
    Float2<Float>* y_2 = (Float2<Float>*) y;
    Float2<Float>* z_2 = (Float2<Float>*) z;
    Float2<Float> a2 = {a[0],a[1]};
    Float2<Float> b2 = {b[0],b[1]};
    Float2<Float> c2 = {c[0],c[1]};
    typedef thrust::device_ptr<Float2<Float> > DevF2;
    DevF2 dev_x_2(x_2);
    DevF2 dev_y_2(y_2);
    DevF2 dev_z_2(z_2);
    typedef thrust::tuple<DevF2,DevF2,DevF2> tplDevF2DevF2DevF2;
    typedef thrust::zip_iterator<tplDevF2DevF2DevF2> zipTplDevF2DevF2DevF2;
    try{
      zipTplDevF2DevF2DevF2 z1 = thrust::make_zip_iterator(thrust::make_tuple(dev_x_2,dev_y_2,dev_z_2));
      zipTplDevF2DevF2DevF2 z2 = thrust::make_zip_iterator(thrust::make_tuple(dev_x_2+NN,dev_y_2+NN,dev_z_2+NN));
      thrust::for_each(z1,z2,Axpbypcz<Float>(a2,b2,c2));
    }
    catch(thrust::system_error &err){
      PLEGMA_error("Thrust error detected: %s", err.what());
    }
  }

  template void axpbypcz<float>(int NN, float a[2], float* x, float b[2], float* y, float c[2], float* z);
  template void axpbypcz<double>(int NN, double a[2], double* x, double b[2], double* y, double c[2], double* z);
}
