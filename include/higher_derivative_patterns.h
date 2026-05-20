
#pragma once

#include <PLEGMA_Correlator.h>
#include "PLEGMA_Correlator_patterns.h"
#include "PLEGMA_QLoops_patterns.h"
#include <PLEGMA_threep.cuh>

#include <array>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

using namespace plegma;

// ------------------------------------------------------------
// 内部编码约定：
//   X,Y,Z  : 空间方向导数
//   T      : 只用于 gamma_t / 逻辑推断，不用于导数 pattern
//   TP, TM : 时间方向导数，分别表示 t+ / t-
//
// 注意：
//   真正数值 kernel 做导数时，TP/TM 最终仍会映射到
//     dir = 3, time_mode = TIME_PLUS / TIME_MINUS
//   但在 HDF5 metadata / group naming 里，我们显式区分
//     TP -> 4
//     TM -> 5
// ------------------------------------------------------------
enum Dir { X = 0, Y = 1, Z = 2, T = 3, TP = 4, TM = 5 };
constexpr std::array<int, 5> DERIV_DIRS = {X, Y, Z, TP, TM};

// 最多 5 个导数；没用的位置填 -1
struct ThrpPattern {
  std::string conf_tag = "Conf0000";   // 顶层分组
  std::string ns_tag   = "Ns0";        // 比如 Ns5, Ns6, Ns7 ...

  GAMMAS gamma = G4;                   // 最终插入 gamma
  int n_deriv = 0;                     // 0..5
  std::array<int, 5> derivs = {{-1, -1, -1, -1, -1}};

  bool oneD_conserved = false;         // 给 oneDC 用
  bool isZfac = false;                 // 是否 zfac 输出
};

// ---------------------------
// 小工具
// ---------------------------
inline const char* gamma_name(GAMMAS g) {
  switch (g) {
    case ONE:  return "ONE";
    case G1:   return "G1";
    case G2:   return "G2";
    case G3:   return "G3";
    case G4:   return "G4";
    case G5:   return "G5";
    case G5G1: return "G5G1";
    case G5G2: return "G5G2";
    case G5G3: return "G5G3";
    case G5G4: return "G5G4";
    case S12:  return "S12";
    case S13:  return "S13";
    case S23:  return "S23";
    case S41:  return "S41";
    case S42:  return "S42";
    case S43:  return "S43";
    default:   return "UNKNOWN";
  }
}

// ------------------------------------------------------------
// HDF5 group naming 用的方向编号：
//   x  -> 0
//   y  -> 1
//   z  -> 2
//   t+ -> 4
//   t- -> 5
//
// 这里故意不把导数时间方向压成 3，避免丢失正反向信息。
// T=3 只保留给 gamma_t 或兜底逻辑，不建议用于导数 pattern。
// ------------------------------------------------------------
inline int dir_group_index(int d) {
  switch (d) {
    case X:  return 0;
    case Y:  return 1;
    case Z:  return 2;
    case TP: return 4;
    case TM: return 5;
    // case T:  return 3;  // 仅兜底；正常导数 pattern 不应出现 T
    default: return -1;
  }
}

// description 里保留真正的 signed 时间方向
inline std::string dir_signed_label(int d) {
  switch (d) {
    case X:  return "x";
    case Y:  return "y";
    case Z:  return "z";
    case T:  return "t";
    case TP: return "t+";
    case TM: return "t-";
    default: return "?";
  }
}

inline std::string deriv_rank_label(const ThrpPattern& pat) {
  if (pat.n_deriv == 0) return "Local";
  if (pat.n_deriv == 1) return pat.oneD_conserved ? "OneDC" : "OneD";
  if (pat.n_deriv == 2) return "TwoD";
  if (pat.n_deriv == 3) return "ThreeD";
  if (pat.n_deriv == 4) return "FourD";
  if (pat.n_deriv == 5) return "FiveD";
  return "higherD";
}

// group 里的末级路径
// 例：
//   n=1 -> dir4      (t+)
//   n=1 -> dir5      (t-)
//   n=2 -> dirs01
//   n=3 -> dirs014
//   n=4 -> dirs0155
inline std::string deriv_group_leaf(const ThrpPattern& pat) {
  if (pat.n_deriv <= 0) return "";

  std::ostringstream os;
  if (pat.n_deriv == 1) {
    os << "dir" << dir_group_index(pat.derivs[0]);
  } else {
    os << "dirs";
    for (int i = 0; i < pat.n_deriv; ++i) {
      os << dir_group_index(pat.derivs[i]);
    }
  }
  return os.str();
}

// description 里保存真正 pattern
// 例如：
// gamma=G2 ; n_deriv=4 ; derivs=t+,y,z,z ; deriv_group=4122
inline std::string make_pattern_description(const ThrpPattern& pat) {
  std::ostringstream os;
  os << "gamma=" << gamma_name(pat.gamma);
  os << " ; n_deriv=" << pat.n_deriv;

  os << " ; derivs=";
  if (pat.n_deriv == 0) {
    os << "(none)";
  } else {
    for (int i = 0; i < pat.n_deriv; ++i) {
      if (i) os << ",";
      os << dir_signed_label(pat.derivs[i]);
    }
  }

  os << " ; deriv_group=";
  if (pat.n_deriv == 0) {
    os << "Local";
  } else {
    for (int i = 0; i < pat.n_deriv; ++i) {
      os << dir_group_index(pat.derivs[i]);
    }
  }

  return os.str();
}

// 最终 group 路径
// 例如：
// Conf0000/Ns6/localLoops
// Conf0000/Ns6/OneD/dir4
// Conf0000/Ns6/TwoD/dirs01
// Conf0000/Ns6/FiveD/dirs40125
inline std::string make_pattern_group(const ThrpPattern& pat) {
  std::ostringstream os;
  // os << pat.conf_tag << "/" << pat.ns_tag << "/" << deriv_rank_label(pat);
  os << deriv_rank_label(pat);

  std::string leaf = deriv_group_leaf(pat);
  if (!leaf.empty()) {
    os << "/" << leaf;
  }
  return os.str();
}

// 目前 dataset 固定叫 loop，和你示例一致
inline std::string make_pattern_dataset(const ThrpPattern&) {
  return "threep";
}


// template<typename Float>
// void prepare_pattern_metadata(PLEGMA_Correlator<Float>& corr,
//                               const ThrpPattern& pat,
//                               bool isZfac = false)
// {
//   corr.preparePatternMetadata(make_pattern_group(pat),
//                               make_pattern_dataset(pat),
//                               make_pattern_description(pat),
//                               isZfac || pat.isZfac);
// }

// 真正做 contraction
// 注意：这里仍然走 threep_local，因为导数已经作用在 fwdProp 上了
template<typename Float>
void contractNucleonThrp_patterns(
    PLEGMA_Correlator_patterns<Float>& corr,
    PLEGMA_Propagator<Float>& bwdProp,
    PLEGMA_Propagator<Float>& fwdProp,
    int signProps,
    const ThrpPattern& pat,
    bool isZfac = false)
{
  // corr.configurePatternMetadata( pat, isZfac || pat.isZfac);
  corr.configurePatternMetadata(make_pattern_group(pat),
                            make_pattern_dataset(pat),
                            make_pattern_description(pat),
                            isZfac || pat.isZfac);

  std::vector<GAMMAS> gammas = {pat.gamma};

  threep_local<true, Float, Float, Float>(
      corr, bwdProp, fwdProp, signProps, gammas, isZfac || pat.isZfac);
}


// contractG5 with pattern metadata for disconnected loops.
// Derivatives are already applied to x_l / x_r before calling.
// All 16 gamma channels are computed in one shot by contractG5.
//
// Templated on QLoopsT so that the call site can pass either
// PLEGMA_QLoops_patterns<Float> (stock, host accumulator) or
// PLEGMA_QLoops_fast<Float> (device accumulator, no D2H per call).
template<typename Float, typename QLoopsT>
void contractG5_patterns(
    QLoopsT& qLoops,
    PLEGMA_Vector<Float>& x_l,
    PLEGMA_Vector<Float>& x_r,
    Float val,
    bool accum,
    const ThrpPattern& pat)
{
  qLoops.configurePatternMetadata(make_pattern_group(pat),
                                  "loop",
                                  make_pattern_description(pat));

  qLoops.oneEnd_trick(x_l, x_r, val, accum);
}

// Multi-isc variant: sums Nsc disconnected-loop contributions on the
// device into qLoops's accumulator (PLEGMA_QLoops_fast::d_acc).
// Does NOT run the FT — the caller is expected to follow up with
//   qLoops.batchStore(ft, accumulate);
// once per pattern.  Compared to calling contractG5_patterns Nsc
// times this saves (Nsc-1) FTs per pattern.
template<typename Float, typename QLoopsT>
void contractG5_patterns_multi(
    QLoopsT& qLoops,
    const std::vector<PLEGMA_Vector<Float>*>& x_l,
    const std::vector<PLEGMA_Vector<Float>*>& x_r,
    const std::vector<Float>& vals,
    const ThrpPattern& pat)
{
  qLoops.configurePatternMetadata(make_pattern_group(pat),
                                  "loop",
                                  make_pattern_description(pat));

  qLoops.oneEnd_trick_fast_multi(x_l, x_r, vals);
}