#include <PLEGMA.h>
#include <color_spinor_field.h>
#include "PLEGMA_vector_utils.cuh"   // 用 PLEGMA 工程里的正确相对路径
using namespace quda;
using namespace plegma;

template <typename Float>
void prop_copyToQUDA_offset(PLEGMA_Propagator<Float> &prop,
                            std::vector<ColorSpinorField> &qudaVec,
                            int rhs_offset, bool isEv)
{
  for (int isc = 0; isc < 12; ++isc) {
    PLEGMA_Vector<Float> tmp;
    tmp.absorb(&prop, isc / 3, isc % 3);          // 直接从 prop 抽出 (nu,c2)
    copy_to_QUDA(tmp.D_elem(), qudaVec, rhs_offset + isc, isEv);
  }
}

template <typename Float>
void prop_copyFromQUDA_offset(PLEGMA_Propagator<Float> &prop,
                              std::vector<ColorSpinorField> &qudaVec,
                              int rhs_offset, bool isEv)
{
  for (int isc = 0; isc < 12; ++isc) {
    PLEGMA_Vector<Float> tmp;
    copy_from_QUDA(tmp.D_elem(), qudaVec, rhs_offset + isc, isEv);
    prop.absorb(tmp, isc / 3, isc % 3);
  }
}

// 显式实例化，保证符号进库
template void prop_copyToQUDA_offset<float>(PLEGMA_Propagator<float>&, std::vector<ColorSpinorField>&, int, bool);
template void prop_copyFromQUDA_offset<float>(PLEGMA_Propagator<float>&, std::vector<ColorSpinorField>&, int, bool);
template void prop_copyToQUDA_offset<double>(PLEGMA_Propagator<double>&, std::vector<ColorSpinorField>&, int, bool);
template void prop_copyFromQUDA_offset<double>(PLEGMA_Propagator<double>&, std::vector<ColorSpinorField>&, int, bool);

template <typename Float>
void vec_copyToQUDA_offset(std::vector<PLEGMA_Vector<Float>*> &vecs,
                           std::vector<ColorSpinorField> &qudaVec,
                           int rhs_offset, bool isEv)
{
  const int n = (int)vecs.size();
  for (int i = 0; i < n; ++i) {
    if (!vecs[i]) PLEGMA_error("vec_copyToQUDA_offset: null vec pointer at i=%d", i);
    copy_to_QUDA(vecs[i]->D_elem(), qudaVec, rhs_offset + i, isEv);
  }
}

template <typename Float>
void vec_copyFromQUDA_offset(std::vector<PLEGMA_Vector<Float>*> &vecs,
                             std::vector<ColorSpinorField> &qudaVec,
                             int rhs_offset, bool isEv)
{
  const int n = (int)vecs.size();
  for (int i = 0; i < n; ++i) {
    if (!vecs[i]) PLEGMA_error("vec_copyFromQUDA_offset: null vec pointer at i=%d", i);
    copy_from_QUDA(vecs[i]->D_elem(), qudaVec, rhs_offset + i, isEv);
  }
}

// Explicit instantiation
template void vec_copyToQUDA_offset<float>(std::vector<PLEGMA_Vector<float>*>&, std::vector<ColorSpinorField>&, int, bool);
template void vec_copyFromQUDA_offset<float>(std::vector<PLEGMA_Vector<float>*>&, std::vector<ColorSpinorField>&, int, bool);
template void vec_copyToQUDA_offset<double>(std::vector<PLEGMA_Vector<double>*>&, std::vector<ColorSpinorField>&, int, bool);
template void vec_copyFromQUDA_offset<double>(std::vector<PLEGMA_Vector<double>*>&, std::vector<ColorSpinorField>&, int, bool);
