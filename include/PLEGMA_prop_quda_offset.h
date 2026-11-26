#pragma once
#include <vector>
#include <color_spinor_field.h>
#include <PLEGMA.h>

template <typename Float>
void prop_copyToQUDA_offset(plegma::PLEGMA_Propagator<Float> &prop,
                            std::vector<quda::ColorSpinorField> &qudaVec,
                            int rhs_offset, bool isEv);

template <typename Float>
void prop_copyFromQUDA_offset(plegma::PLEGMA_Propagator<Float> &prop,
                              std::vector<quda::ColorSpinorField> &qudaVec,
                              int rhs_offset, bool isEv);

template <typename Float>
void vec_copyToQUDA_offset(std::vector<plegma::PLEGMA_Vector<Float>*> &vecs,
                           std::vector<quda::ColorSpinorField> &qudaVec,
                           int rhs_offset, bool isEv);

template <typename Float>
void vec_copyFromQUDA_offset(std::vector<plegma::PLEGMA_Vector<Float>*> &vecs,
                             std::vector<quda::ColorSpinorField> &qudaVec,
                             int rhs_offset, bool isEv);
