#include <PLEGMA_utils.h>
#include "invert_quda.h"
#include <assert.h>
#include "util_quda.h"

inline QudaVerbosity get_verbosity_type(std::string s) {
  if(s=="silent")
    return QUDA_SILENT;
  else if(s=="summarize")
    return QUDA_SUMMARIZE;
  else if(s=="verbose")
    return QUDA_VERBOSE;
  else if(s=="debug")
    return QUDA_DEBUG_VERBOSE;
  else {
    PLEGMA_warning("invalid verbosity type %s\n", s.c_str());
    return QUDA_INVALID_VERBOSITY;
  }
}

inline std::string get_verbosity_str(QudaVerbosity type) {
  switch(type) {
  case QUDA_SILENT:
    return "silent";
  case QUDA_SUMMARIZE:
    return "summarize";
  case QUDA_VERBOSE:
    return "verbose";
  case QUDA_DEBUG_VERBOSE:
    return "debug";
  default:
    PLEGMA_warning("invalid verbosity type %d\n", type);
    return "invalid";
  }
}

inline QudaReconstructType get_recon(std::string s) {
  if(s=="8")
    return  QUDA_RECONSTRUCT_8;
  else if(s=="9")
    return  QUDA_RECONSTRUCT_9;
  else if(s=="12")
    return  QUDA_RECONSTRUCT_12;
  else if(s=="13")
    return  QUDA_RECONSTRUCT_13;
  else if(s=="18")
    return  QUDA_RECONSTRUCT_NO;
  else {
    PLEGMA_warning( "Error: invalid reconstruct type %s\n", s.c_str());
    return QUDA_RECONSTRUCT_INVALID;
  }
}

inline std::string get_recon_str(QudaReconstructType recon) {
  switch(recon){
  case QUDA_RECONSTRUCT_13:
    return"13";
  case QUDA_RECONSTRUCT_12:
    return"12";
  case QUDA_RECONSTRUCT_9:
    return"9";
  case QUDA_RECONSTRUCT_8:
    return "8";
  case QUDA_RECONSTRUCT_NO:
    return "18";
  default:
    PLEGMA_warning( "Error: invalid reconstruct type %d\n", recon);
    return"invalid";	
  }
}

inline QudaPrecision get_prec(std::string s) {
  if(s=="double")
    return QUDA_DOUBLE_PRECISION;
  else if(s=="single")
    return QUDA_SINGLE_PRECISION;
  else if(s=="half")
    return QUDA_HALF_PRECISION;
  else if(s=="half")
    return QUDA_HALF_PRECISION;
  else if(s=="quarter")
    return QUDA_QUARTER_PRECISION;
  else {
    PLEGMA_warning( "invalid precision type %s\n",s.c_str());
    return QUDA_INVALID_PRECISION;
  }
}

inline std::string get_prec_str(QudaPrecision prec) {
  switch (prec) {
  case QUDA_DOUBLE_PRECISION:
    return "double";
  case QUDA_SINGLE_PRECISION:
    return"single";
  case QUDA_HALF_PRECISION:
    return"half";
  case QUDA_QUARTER_PRECISION:
    return"quarter";
  default:
    PLEGMA_warning( "invalid precision type %d\n", prec);
    return "invalid";
  }
}

inline std::string get_gauge_order_str(QudaGaugeFieldOrder order) {
  switch(order){
  case QUDA_QDP_GAUGE_ORDER:
    return "qdp";
  case QUDA_MILC_GAUGE_ORDER:
    return "milc";
  case QUDA_CPS_WILSON_GAUGE_ORDER:
    return "cps_wilson";
  default:
    PLEGMA_warning( "invalid gauge order type %d\n", order);
    return "invalid";
  }	
}

inline QudaDslashType get_dslash_type(std::string s) {
  if(s=="wilson")
    return QUDA_WILSON_DSLASH;
  else if(s=="clover")
    return QUDA_CLOVER_WILSON_DSLASH;
  else if(s=="twisted-mass")
    return QUDA_TWISTED_MASS_DSLASH;
  else if(s=="twisted-clover")
    return QUDA_TWISTED_CLOVER_DSLASH;
  else if(s=="staggered")
    return  QUDA_STAGGERED_DSLASH;
  else if(s=="asqtad")
    return  QUDA_ASQTAD_DSLASH;
  else if(s=="domain-wall")
    return  QUDA_DOMAIN_WALL_DSLASH;
  else if(s=="domain-wall-4d")
    return  QUDA_DOMAIN_WALL_4D_DSLASH;
  else if(s=="mobius")
    return  QUDA_MOBIUS_DWF_DSLASH;
  else if(s=="laplace")
    return  QUDA_LAPLACE_DSLASH;
  else {
    PLEGMA_warning( "invalid dslash type %s\n", s.c_str());	
    return QUDA_INVALID_DSLASH;
  }
}

inline std::string get_dslash_str(QudaDslashType type) {
  switch(type){	
  case QUDA_WILSON_DSLASH:
    return "wilson";
  case QUDA_CLOVER_WILSON_DSLASH:
    return"clover";
  case QUDA_TWISTED_MASS_DSLASH:
    return"twisted-mass";
  case QUDA_TWISTED_CLOVER_DSLASH:
    return"twisted-clover";
  case QUDA_STAGGERED_DSLASH:
    return "staggered";
  case QUDA_ASQTAD_DSLASH:
    return "asqtad";
  case QUDA_DOMAIN_WALL_DSLASH:
    return "domain-wall";
  case QUDA_DOMAIN_WALL_4D_DSLASH:
    return "domain_wall_4d";
  case QUDA_MOBIUS_DWF_DSLASH:
    return "mobius";
  case QUDA_LAPLACE_DSLASH:
    return "laplace";
  default:
    PLEGMA_warning( "invalid dslash type %d\n", type); 
    return "invalid";
  }
}

inline QudaMassNormalization get_mass_normalization_type(std::string s) {
  if(s=="kappa")
    return QUDA_KAPPA_NORMALIZATION;
  else if(s=="mass")
    return QUDA_MASS_NORMALIZATION;
  else if(s=="asym-mass")
    return QUDA_ASYMMETRIC_MASS_NORMALIZATION;
  else {
    PLEGMA_warning( "invalid mass normalization %s\n", s.c_str());
    return QUDA_INVALID_NORMALIZATION;
  }
}

inline std::string get_mass_normalization_str(QudaMassNormalization type) {
  switch (type) {
  case QUDA_KAPPA_NORMALIZATION:
    return "kappa";
  case QUDA_MASS_NORMALIZATION:
    return "mass";
  case QUDA_ASYMMETRIC_MASS_NORMALIZATION:
    return "asym-mass";
  default:
    PLEGMA_warning( "invalid mass normalization\n");
    return "invalid";
  }
}

inline QudaMatPCType get_matpc_type(std::string s) {
  if(s=="even-even")
    return QUDA_MATPC_EVEN_EVEN;
  else if(s=="odd-odd")
    return QUDA_MATPC_ODD_ODD;
  else if(s=="even-even-asym")
    return QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  else if(s=="odd-odd-asym")
    return QUDA_MATPC_ODD_ODD_ASYMMETRIC;
  else {
    PLEGMA_warning( "invalid matpc type %s\n", s.c_str());
    return QUDA_MATPC_INVALID;
  }
}

inline std::string get_matpc_str(QudaMatPCType type) {
  switch(type) {
  case QUDA_MATPC_EVEN_EVEN:
    return "even-even";
  case QUDA_MATPC_ODD_ODD:
    return "odd-odd";
  case QUDA_MATPC_EVEN_EVEN_ASYMMETRIC:
    return "even-even-asym";
  case QUDA_MATPC_ODD_ODD_ASYMMETRIC:
    return "odd-odd-asym";
  default:
    PLEGMA_warning( "invalid matpc type %d\n", type);
    return "invalid";
  }
}

inline QudaSolveType get_solve_type(std::string s) {
  if(s=="direct")
    return QUDA_DIRECT_SOLVE;
  else if(s=="direct-pc")
    return QUDA_DIRECT_PC_SOLVE;
  else if(s=="normop")
    return QUDA_NORMOP_SOLVE;
  else if(s=="normop-pc")
    return QUDA_NORMOP_PC_SOLVE;
  else if(s=="normerr")
    return QUDA_NORMERR_SOLVE;
  else if(s=="normerr-pc")
    return QUDA_NORMERR_PC_SOLVE;
  else {
    PLEGMA_warning( "invalid matpc type %s\n", s.c_str());
    return QUDA_INVALID_SOLVE;
  }
}

inline std::string get_solve_str(QudaSolveType type) {
  switch(type) {
  case QUDA_DIRECT_SOLVE:
    return "direct";
  case QUDA_DIRECT_PC_SOLVE:
    return "direct-pc";
  case QUDA_NORMOP_SOLVE:
    return "normop";
  case QUDA_NORMOP_PC_SOLVE:
    return "normop-pc";
  case QUDA_NORMERR_SOLVE:
    return "normerr";
  case QUDA_NORMERR_PC_SOLVE:
    return "normerr-pc";
  default:
    PLEGMA_warning( "invalid solve type %d\n", type);
    return "invalid";
  }
}

inline QudaSchwarzType get_schwarz_type(std::string s) {
  if(s=="false")
    return QUDA_INVALID_SCHWARZ;
  else if(s=="add")
    return QUDA_ADDITIVE_SCHWARZ;
  else if(s=="mul")
    return QUDA_MULTIPLICATIVE_SCHWARZ;
  else {
    PLEGMA_warning( "invalid Schwarz type %s\n", s.c_str());
    return QUDA_INVALID_SCHWARZ;
  }
}

inline QudaTwistFlavorType get_flavor_type(std::string s) {
  if(s=="singlet")
    return QUDA_TWIST_SINGLET;
  else if(s=="deg-doublet")
    return QUDA_TWIST_DEG_DOUBLET;
  else if(s=="nondeg-doublet")
    return QUDA_TWIST_NONDEG_DOUBLET;
  else if(s=="no")
    return  QUDA_TWIST_NO;
  else {
    PLEGMA_warning( "invalid flavor type %s\n", s.c_str());	
    return QUDA_TWIST_INVALID;
  }
}

inline std::string get_flavor_str(QudaTwistFlavorType type) {
  switch(type) {
  case QUDA_TWIST_SINGLET:
    return "singlet";
  case QUDA_TWIST_DEG_DOUBLET:
    return "deg-doublet";
  case QUDA_TWIST_NONDEG_DOUBLET:
    return "nondeg-doublet";
  case QUDA_TWIST_NO:
    return "no";
  default:
    PLEGMA_warning( "invalid flavor type %d\n", type);	
    return "invalid";
  }
}

inline QudaInverterType get_solver_type(std::string s) {
  if(s=="cg")
    return QUDA_CG_INVERTER;
  else if(s=="bicgstab")
    return QUDA_BICGSTAB_INVERTER;
  else if(s=="gcr")
    return QUDA_GCR_INVERTER;
  else if(s=="pcg")
    return QUDA_PCG_INVERTER;
  else if(s=="mpcg")
    return QUDA_MPCG_INVERTER; 
  else if(s=="mpbicgstab")
    return QUDA_MPBICGSTAB_INVERTER;
  else if(s=="mr")
    return QUDA_MR_INVERTER;
  else if(s=="sd")
    return QUDA_SD_INVERTER;
  else if(s=="eigcg")
    return QUDA_EIGCG_INVERTER;
  else if(s=="inc-eigcg")
    return QUDA_INC_EIGCG_INVERTER;
  else if(s=="gmresdr")
    return QUDA_GMRESDR_INVERTER;
  else if(s=="gmresdr-proj")
    return QUDA_GMRESDR_PROJ_INVERTER;
  else if(s=="gmresdr-sh")
    return QUDA_GMRESDR_SH_INVERTER;
  else if(s=="fgmresdr")
    return QUDA_FGMRESDR_INVERTER;
  else if(s=="mg")
    return QUDA_MG_INVERTER;
  else if(s=="bicgstab-l")
    return QUDA_BICGSTABL_INVERTER;
  else if(s=="cgne")
    return QUDA_CGNE_INVERTER;
  else if(s=="cgnr")
    return QUDA_CGNR_INVERTER;
  else if(s=="cg3")
    return QUDA_CG3_INVERTER;
  else if(s=="cg3ne")
    return QUDA_CG3NE_INVERTER;
  else if(s=="cg3nr")
    return QUDA_CG3NR_INVERTER;
  else if(s=="ca-cg")
    return QUDA_CA_CG_INVERTER;
  else if(s=="ca-gcr")
    return QUDA_CA_GCR_INVERTER;
  else if(s=="none")
    return QUDA_INVALID_INVERTER;
  else {
    PLEGMA_warning( "invalid solver type %s\n", s.c_str());
    return QUDA_INVALID_INVERTER;
  }
}

inline std::string get_solver_str(QudaInverterType type) {
  switch(type){
  case QUDA_CG_INVERTER:
    return "cg";
  case QUDA_BICGSTAB_INVERTER:
    return "bicgstab";
  case QUDA_GCR_INVERTER:
    return "gcr";
  case QUDA_PCG_INVERTER:
    return "pcg";
  case QUDA_MPCG_INVERTER:
    return "mpcg";
  case QUDA_MPBICGSTAB_INVERTER:
    return "mpbicgstab";
  case QUDA_MR_INVERTER:
    return "mr";
  case QUDA_SD_INVERTER:
    return "sd";
  case QUDA_EIGCG_INVERTER:
    return "eigcg";
  case QUDA_INC_EIGCG_INVERTER:
    return "inc-eigcg";
  case QUDA_GMRESDR_INVERTER:
    return "gmresdr";
  case QUDA_GMRESDR_PROJ_INVERTER:
    return "gmresdr-proj";
  case QUDA_GMRESDR_SH_INVERTER:
    return "gmresdr-sh";
  case QUDA_FGMRESDR_INVERTER:
    return "fgmresdr";
  case QUDA_MG_INVERTER:
    return"mg";
  case QUDA_BICGSTABL_INVERTER:
    return "bicgstab-l";
  case QUDA_CGNE_INVERTER:
    return "cgne";
  case QUDA_CGNR_INVERTER:
    return "cgnr";
  case QUDA_CG3_INVERTER:
    return "cg3";
  case QUDA_CG3NE_INVERTER:
    return "cg3ne";
  case QUDA_CG3NR_INVERTER:
    return "cg3nr";
  case QUDA_CA_CG_INVERTER:
    return "ca-cg";
  case QUDA_CA_GCR_INVERTER:
    return "ca-gcr";
  case QUDA_INVALID_INVERTER:
    return "none";
  default:
    PLEGMA_warning("invalid solver type %d\n", type);
    return "";
  }
}

inline QudaBoolean get_boolean(std::string s){
  if(s.compare("true")==0) return QUDA_BOOLEAN_YES;
  else if (s.compare("false")==0) return QUDA_BOOLEAN_NO;
  PLEGMA_warning("Boolean not recognized");
  return QUDA_BOOLEAN_YES;
}


inline QudaEigSpectrumType get_eigensolution_type(std::string s) {
  if(s=="sr")
    return QUDA_SPECTRUM_SR_EIG;
  else if(s=="lr")
    return QUDA_SPECTRUM_LR_EIG;
  else if(s=="sm")
    return QUDA_SPECTRUM_SM_EIG;
  else if(s=="lm")
    return QUDA_SPECTRUM_LM_EIG;
  else if(s=="si")
    return QUDA_SPECTRUM_SI_EIG; 
  else if(s=="li")
    return QUDA_SPECTRUM_LI_EIG; 
  else {
    PLEGMA_warning( "invalid eigensolver type %s\n", s.c_str());
    return QUDA_SPECTRUM_INVALID;
  }
}

inline  QudaEigType get_eigensolver(std::string s){
  if(s=="trlan" || s=="trlm")
    return QUDA_EIG_TR_LANCZOS;
  else if(s=="irlan" || s=="irlm")
    return QUDA_EIG_IR_LANCZOS;
  else if(s=="irarn" || s=="iram")
    return QUDA_EIG_IR_ARNOLDI;
  else {
    PLEGMA_warning("Invalid eigensolver for multigrid exact deflation\n");
    return QUDA_EIG_INVALID;
  }
}


