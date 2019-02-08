#include <PLEGMA_utils.h>
#include "invert_quda.h"
#include <assert.h>
#include "util_quda.h"

QudaVerbosity
get_verbosity_type(const char* s)
{
  QudaVerbosity ret =  QUDA_INVALID_VERBOSITY;

  if (strcmp(s, "silent") == 0){
    ret = QUDA_SILENT;
  }else if (strcmp(s, "summarize") == 0){
    ret = QUDA_SUMMARIZE;
  }else if (strcmp(s, "verbose") == 0){
    ret = QUDA_VERBOSE;
  }else if (strcmp(s, "debug") == 0){
    ret = QUDA_DEBUG_VERBOSE;
  }else{
    fprintf(stderr, "Error: invalid verbosity type %s\n", s);
    exit(1);
  }

  return ret;
}

const char *
get_verbosity_str(QudaVerbosity type)
{
  const char* ret;

  switch(type) {
  case QUDA_SILENT:
    ret = "silent";
    break;
  case QUDA_SUMMARIZE:
    ret = "summarize";
    break;
  case QUDA_VERBOSE:
    ret = "verbose";
    break;
  case QUDA_DEBUG_VERBOSE:
    ret = "debug";
    break;
  default:
    fprintf(stderr, "Error: invalid verbosity type %d\n", type);
    exit(1);
  }

  return ret;
}

QudaReconstructType
get_recon(const char* s)
{
    QudaReconstructType  ret;
    
    if (strcmp(s, "8") == 0){
	ret =  QUDA_RECONSTRUCT_8;
    }else if (strcmp(s, "9") == 0){
	ret =  QUDA_RECONSTRUCT_9;
    }else if (strcmp(s, "12") == 0){
	ret =  QUDA_RECONSTRUCT_12;
    }else if (strcmp(s, "13") == 0){
	ret =  QUDA_RECONSTRUCT_13;
    }else if (strcmp(s, "18") == 0){
	ret =  QUDA_RECONSTRUCT_NO;
    }else{
	fprintf(stderr, "Error: invalid reconstruct type\n");
	exit(1);
    }
    
    return ret;
    
    
}

QudaPrecision
get_prec(const char* s)
{
  QudaPrecision ret = QUDA_DOUBLE_PRECISION;

  if (strcmp(s, "double") == 0) {
    ret = QUDA_DOUBLE_PRECISION;
  } else if (strcmp(s, "single") == 0) {
    ret = QUDA_SINGLE_PRECISION;
  } else if (strcmp(s, "half") == 0) {
    ret = QUDA_HALF_PRECISION;
  } else if (strcmp(s, "half") == 0) {
    ret = QUDA_HALF_PRECISION;
  } else if (strcmp(s, "quarter") == 0) {
    ret = QUDA_QUARTER_PRECISION;
  } else {
    fprintf(stderr, "Error: invalid precision type\n");
    exit(1);
  }

  return ret;
}

const char*
get_prec_str(QudaPrecision prec)
{
  const char* ret;

  switch (prec) {
  case QUDA_DOUBLE_PRECISION:
    ret=  "double";
    break;
  case QUDA_SINGLE_PRECISION:
    ret= "single";
    break;
  case QUDA_HALF_PRECISION:
    ret= "half";
    break;
  case QUDA_QUARTER_PRECISION:
    ret= "quarter";
    break;
  default:
    ret = "unknown";
    break;
  }

  return ret;
}


const char* 
get_unitarization_str(bool svd_only)
{
  const char* ret;
 
  if(svd_only){
    ret = "SVD";
  }else{
    ret = "Cayley-Hamilton/SVD";
  }
  return ret;
}

const char* 
get_gauge_order_str(QudaGaugeFieldOrder order)
{
  const char* ret;

  switch(order){
    case QUDA_QDP_GAUGE_ORDER:
	ret = "qdp";
	break;

    case QUDA_MILC_GAUGE_ORDER:
	ret = "milc";
	break;

    case QUDA_CPS_WILSON_GAUGE_ORDER:
	ret = "cps_wilson";
	break;

    default:
	ret = "unknown";
	break;
  }	

  return ret;
}


const char* 
get_recon_str(QudaReconstructType recon)
{
    const char* ret;
    switch(recon){
    case QUDA_RECONSTRUCT_13:
        ret="13";
        break;
    case QUDA_RECONSTRUCT_12:
	ret= "12";
	break;
    case QUDA_RECONSTRUCT_9:
        ret="9";
        break;
    case QUDA_RECONSTRUCT_8:
	ret = "8";
	break;
    case QUDA_RECONSTRUCT_NO:
	ret = "18";
	break;
    default:
	ret="unknown";
	break;
    }
    
    return ret;
}

const char*
get_test_type(int t)
{
    const char* ret;
    switch(t){
    case 0:
	ret = "even";
	break;
    case 1:
	ret = "odd";
	break;
    case 2:
	ret = "full";
	break;
    case 3:
	ret = "mcg_even";
	break;	
    case 4:
	ret = "mcg_odd";
	break;	
    case 5:
	ret = "mcg_full";
	break;	
    default:
	ret = "unknown";
	break;
    }
    
    return ret;
}

const char*
get_staggered_test_type(int t)
{
    const char* ret;
    switch(t){
    case 0:
  ret = "full";
  break;
    case 1:
  ret = "full_ee_prec";
  break;
    case 2:
  ret = "full_oo_prec";
  break;
    case 3:
  ret = "even";
  break;
    case 4:
  ret = "odd";
  break;
    case 5:
  ret = "mcg_even";
  break;  
    case 6:
  ret = "mcg_odd";
  break;  
    default:
  ret = "unknown";
  break;
    }
    
    return ret;
}

int get_rank_order(const char* s)
{
  int ret = -1;

  if (strcmp(s, "col") == 0) {
    ret = 0;
  } else if (strcmp(s, "row") == 0) {
    ret = 1;
  } else {
    fprintf(stderr, "Error: invalid rank order type\n");
    exit(1);
  }

  return ret;
}

QudaDslashType
get_dslash_type(const char* s)
{
  QudaDslashType ret =  QUDA_INVALID_DSLASH;
  
  if (strcmp(s, "wilson") == 0){
    ret = QUDA_WILSON_DSLASH;
  }else if (strcmp(s, "clover") == 0){
    ret = QUDA_CLOVER_WILSON_DSLASH;
  }else if (strcmp(s, "twisted-mass") == 0){
    ret = QUDA_TWISTED_MASS_DSLASH;
  }else if (strcmp(s, "twisted-clover") == 0){
    ret = QUDA_TWISTED_CLOVER_DSLASH;
  }else if (strcmp(s, "staggered") == 0){
    ret =  QUDA_STAGGERED_DSLASH;
  }else if (strcmp(s, "asqtad") == 0){
    ret =  QUDA_ASQTAD_DSLASH;
  }else if (strcmp(s, "domain-wall") == 0){
    ret =  QUDA_DOMAIN_WALL_DSLASH;
  }else if (strcmp(s, "domain-wall-4d") == 0){
    ret =  QUDA_DOMAIN_WALL_4D_DSLASH;
  }else if (strcmp(s, "mobius") == 0){
    ret =  QUDA_MOBIUS_DWF_DSLASH;
  }else if (strcmp(s, "laplace") == 0){
    ret =  QUDA_LAPLACE_DSLASH;
  }else{
    fprintf(stderr, "Error: invalid dslash type\n");	
    exit(1);
  }
  
  return ret;
}

const char* 
get_dslash_str(QudaDslashType type)
{
  const char* ret;
  
  switch( type){	
  case QUDA_WILSON_DSLASH:
    ret=  "wilson";
    break;
  case QUDA_CLOVER_WILSON_DSLASH:
    ret= "clover";
    break;
  case QUDA_TWISTED_MASS_DSLASH:
    ret= "twisted-mass";
    break;
  case QUDA_TWISTED_CLOVER_DSLASH:
    ret= "twisted-clover";
    break;
  case QUDA_STAGGERED_DSLASH:
    ret = "staggered";
    break;
  case QUDA_ASQTAD_DSLASH:
    ret = "asqtad";
    break;
  case QUDA_DOMAIN_WALL_DSLASH:
    ret = "domain-wall";
    break;
  case QUDA_DOMAIN_WALL_4D_DSLASH:
    ret = "domain_wall_4d";
    break;
  case QUDA_MOBIUS_DWF_DSLASH:
    ret = "mobius";
    break;
  case QUDA_LAPLACE_DSLASH:
    ret = "laplace";
    break;
  default:
    ret = "unknown";	
    break;
  }
  
  
  return ret;
    
}

QudaMassNormalization
get_mass_normalization_type(const char* s)
{
  QudaMassNormalization ret =  QUDA_INVALID_NORMALIZATION;

  if (strcmp(s, "kappa") == 0){
    ret = QUDA_KAPPA_NORMALIZATION;
  }else if (strcmp(s, "mass") == 0){
    ret = QUDA_MASS_NORMALIZATION;
  }else if (strcmp(s, "asym-mass") == 0){
    ret = QUDA_ASYMMETRIC_MASS_NORMALIZATION;
  }else{
    fprintf(stderr, "Error: invalid mass normalization\n");
    exit(1);
  }

  return ret;
}

const char*
get_mass_normalization_str(QudaMassNormalization type)
{
  const char *s;

  switch (type) {
  case QUDA_KAPPA_NORMALIZATION:
    s = "kappa";
    break;
  case QUDA_MASS_NORMALIZATION:
    s = "mass";
    break;
  case QUDA_ASYMMETRIC_MASS_NORMALIZATION:
    s = "asym-mass";
    break;
  default:
    fprintf(stderr, "Error: invalid mass normalization\n");
    exit(1);
  }

  return s;
}

QudaMatPCType
get_matpc_type(const char* s)
{
  QudaMatPCType ret =  QUDA_MATPC_INVALID;

  if (strcmp(s, "even-even") == 0){
    ret = QUDA_MATPC_EVEN_EVEN;
  }else if (strcmp(s, "odd-odd") == 0){
    ret = QUDA_MATPC_ODD_ODD;
  }else if (strcmp(s, "even-even-asym") == 0){
    ret = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  }else if (strcmp(s, "odd-odd-asym") == 0){
    ret = QUDA_MATPC_ODD_ODD_ASYMMETRIC;
  }else{
    fprintf(stderr, "Error: invalid matpc type %s\n", s);
    exit(1);
  }

  return ret;
}

const char *
get_matpc_str(QudaMatPCType type)
{
  const char* ret;

  switch(type) {
  case QUDA_MATPC_EVEN_EVEN:
    ret = "even-even";
    break;
  case QUDA_MATPC_ODD_ODD:
    ret = "odd-odd";
    break;
  case QUDA_MATPC_EVEN_EVEN_ASYMMETRIC:
    ret = "even-even-asym";
    break;
  case QUDA_MATPC_ODD_ODD_ASYMMETRIC:
    ret = "odd-odd-asym";
    break;
  default:
    fprintf(stderr, "Error: invalid matpc type %d\n", type);
    exit(1);
  }

  return ret;
}

QudaSolveType
get_solve_type(const char* s)
{
  QudaSolveType ret = QUDA_INVALID_SOLVE;

  if (strcmp(s, "direct") == 0) {
    ret = QUDA_DIRECT_SOLVE;
  } else if (strcmp(s, "direct-pc") == 0) {
    ret = QUDA_DIRECT_PC_SOLVE;
  } else if (strcmp(s, "normop") == 0) {
    ret = QUDA_NORMOP_SOLVE;
  } else if (strcmp(s, "normop-pc") == 0) {
    ret = QUDA_NORMOP_PC_SOLVE;
  } else if (strcmp(s, "normerr") == 0) {
    ret = QUDA_NORMERR_SOLVE;
  } else if (strcmp(s, "normerr-pc") == 0) {
    ret = QUDA_NORMERR_PC_SOLVE;
  } else {
    fprintf(stderr, "Error: invalid matpc type %s\n", s);
    exit(1);
  }

  return ret;
}

const char *
get_solve_str(QudaSolveType type)
{
  const char* ret;

  switch(type) {
  case QUDA_DIRECT_SOLVE:
    ret = "direct";
    break;
  case QUDA_DIRECT_PC_SOLVE:
    ret = "direct-pc";
    break;
  case QUDA_NORMOP_SOLVE:
    ret = "normop";
    break;
  case QUDA_NORMOP_PC_SOLVE:
    ret = "normop-pc";
    break;
  case QUDA_NORMERR_SOLVE:
    ret = "normerr";
    break;
  case QUDA_NORMERR_PC_SOLVE:
    ret = "normerr-pc";
    break;
  default:
    fprintf(stderr, "Error: invalid solve type %d\n", type);
    exit(1);
  }

  return ret;
}

QudaSchwarzType
get_schwarz_type(const char* s)
{
  QudaSchwarzType ret = QUDA_INVALID_SCHWARZ;

  if (strcmp(s, "false") == 0) {
    ret = QUDA_INVALID_SCHWARZ;
  } else if (strcmp(s, "add") == 0) {
    ret = QUDA_ADDITIVE_SCHWARZ;
  } else if (strcmp(s, "mul") == 0) {
    ret = QUDA_MULTIPLICATIVE_SCHWARZ;
  } else {
    fprintf(stderr, "Error: invalid Schwarz type %s\n", s);
    exit(1);
  }

  return ret;
}

QudaTwistFlavorType
get_flavor_type(const char* s)
{
  QudaTwistFlavorType ret =  QUDA_TWIST_SINGLET;
  
  if (strcmp(s, "singlet") == 0){
    ret = QUDA_TWIST_SINGLET;
  }else if (strcmp(s, "deg-doublet") == 0){
    ret = QUDA_TWIST_DEG_DOUBLET;
  }else if (strcmp(s, "nondeg-doublet") == 0){
    ret = QUDA_TWIST_NONDEG_DOUBLET;
  }else if (strcmp(s, "no") == 0){
    ret =  QUDA_TWIST_NO;
  }else{
    fprintf(stderr, "Error: invalid flavor type\n");	
    exit(1);
  }
  
  return ret;
}

const char*
get_flavor_str(QudaTwistFlavorType type)
{
  const char* ret;
  
  switch(type) {
  case QUDA_TWIST_SINGLET:
    ret = "singlet";
    break;
  case QUDA_TWIST_DEG_DOUBLET:
    ret = "deg-doublet";
    break;
  case QUDA_TWIST_NONDEG_DOUBLET:
    ret = "nondeg-doublet";
    break;
  case QUDA_TWIST_NO:
    ret = "no";
    break;
  default:
    ret = "unknown";
    break;
  }

  return ret;
}

QudaInverterType
get_solver_type(const char* s)
{
  QudaInverterType ret =  QUDA_INVALID_INVERTER;
  
  if (strcmp(s, "cg") == 0){
    ret = QUDA_CG_INVERTER;
  } else if (strcmp(s, "bicgstab") == 0){
    ret = QUDA_BICGSTAB_INVERTER;
  } else if (strcmp(s, "gcr") == 0){
    ret = QUDA_GCR_INVERTER;
  } else if (strcmp(s, "pcg") == 0){
    ret = QUDA_PCG_INVERTER;
  } else if (strcmp(s, "mpcg") == 0){
    ret = QUDA_MPCG_INVERTER; 
  } else if (strcmp(s, "mpbicgstab") == 0){
    ret = QUDA_MPBICGSTAB_INVERTER;
  } else if (strcmp(s, "mr") == 0){
    ret = QUDA_MR_INVERTER;
  } else if (strcmp(s, "sd") == 0){
    ret = QUDA_SD_INVERTER;
  } else if (strcmp(s, "eigcg") == 0){
    ret = QUDA_EIGCG_INVERTER;
  } else if (strcmp(s, "inc-eigcg") == 0){
    ret = QUDA_INC_EIGCG_INVERTER;
  } else if (strcmp(s, "gmresdr") == 0){
    ret = QUDA_GMRESDR_INVERTER;
  } else if (strcmp(s, "gmresdr-proj") == 0){
    ret = QUDA_GMRESDR_PROJ_INVERTER;
  } else if (strcmp(s, "gmresdr-sh") == 0){
    ret = QUDA_GMRESDR_SH_INVERTER;
  } else if (strcmp(s, "fgmresdr") == 0){
    ret = QUDA_FGMRESDR_INVERTER;
  } else if (strcmp(s, "mg") == 0){
    ret = QUDA_MG_INVERTER;
  } else if (strcmp(s, "bicgstab-l") == 0){
    ret = QUDA_BICGSTABL_INVERTER;
  } else if (strcmp(s, "cgne") == 0){
    ret = QUDA_CGNE_INVERTER;
  } else if (strcmp(s, "cgnr") == 0){
    ret = QUDA_CGNR_INVERTER;
  } else if (strcmp(s, "cg3") == 0){
    ret = QUDA_CG3_INVERTER;
  } else if (strcmp(s, "cg3ne") == 0){
    ret = QUDA_CG3NE_INVERTER;
  } else if (strcmp(s, "cg3nr") == 0){
    ret = QUDA_CG3NR_INVERTER;
  } else if (strcmp(s, "ca-cg") == 0){
    ret = QUDA_CA_CG_INVERTER;
  } else if (strcmp(s, "ca-gcr") == 0){
    ret = QUDA_CA_GCR_INVERTER;
  } else {
    fprintf(stderr, "Error: invalid solver type %s\n", s);
    exit(1);
  }
  
  return ret;
}

const char* 
get_solver_str(QudaInverterType type)
{
  const char* ret;
  
  switch(type){
  case QUDA_CG_INVERTER:
    ret = "cg";
    break;
  case QUDA_BICGSTAB_INVERTER:
    ret = "bicgstab";
    break;
  case QUDA_GCR_INVERTER:
    ret = "gcr";
    break;
  case QUDA_PCG_INVERTER:
    ret = "pcg";
    break;
  case QUDA_MPCG_INVERTER:
    ret = "mpcg";
    break;
  case QUDA_MPBICGSTAB_INVERTER:
    ret = "mpbicgstab";
    break;
  case QUDA_MR_INVERTER:
    ret = "mr";
    break;
  case QUDA_SD_INVERTER:
    ret = "sd";
    break;
  case QUDA_EIGCG_INVERTER:
    ret = "eigcg";
    break;
  case QUDA_INC_EIGCG_INVERTER:
    ret = "inc-eigcg";
    break;
  case QUDA_GMRESDR_INVERTER:
    ret = "gmresdr";
    break;
  case QUDA_GMRESDR_PROJ_INVERTER:
    ret = "gmresdr-proj";
    break;
  case QUDA_GMRESDR_SH_INVERTER:
    ret = "gmresdr-sh";
    break;
  case QUDA_FGMRESDR_INVERTER:
    ret = "fgmresdr";
    break;
  case QUDA_MG_INVERTER:
    ret= "mg";
    break;
  case QUDA_BICGSTABL_INVERTER:
    ret = "bicgstab-l";
    break;
  case QUDA_CGNE_INVERTER:
    ret = "cgne";
    break;
  case QUDA_CGNR_INVERTER:
    ret = "cgnr";
    break;
  case QUDA_CG3_INVERTER:
    ret = "cg3";
    break;
  case QUDA_CG3NE_INVERTER:
    ret = "cg3ne";
    break;
  case QUDA_CG3NR_INVERTER:
    ret = "cg3nr";
    break;
  case QUDA_CA_CG_INVERTER:
    ret = "ca-cg";
    break;
  case QUDA_CA_GCR_INVERTER:
    ret = "ca-gcr";
    break;
  default:
    ret = "unknown";
    errorQuda("Error: invalid solver type %d\n", type);
    break;
  }

  return ret;
}

const char* 
get_quda_ver_str()
{
  static char vstr[32];
  int major_num = QUDA_VERSION_MAJOR;
  int minor_num = QUDA_VERSION_MINOR;
  int ext_num = QUDA_VERSION_SUBMINOR;
  sprintf(vstr, "%1d.%1d.%1d", 
	  major_num,
	  minor_num,
	  ext_num);
  return vstr;
}


QudaExtLibType
get_solve_ext_lib_type(const char* s)
{
  QudaExtLibType ret = QUDA_EXTLIB_INVALID;

  if (strcmp(s, "eigen") == 0) {
    ret = QUDA_EIGEN_EXTLIB;
  } else if (strcmp(s, "magma") == 0) {
    ret = QUDA_MAGMA_EXTLIB;
  } else {
    fprintf(stderr, "Error: invalid external library type %s\n", s);
    exit(1);
  }

  return ret;
}

QudaFieldLocation
get_location(const char* s)
{
  QudaFieldLocation ret = QUDA_INVALID_FIELD_LOCATION;

  if (strcmp(s, "cpu") == 0 || strcmp(s, "host") == 0) {
    ret = QUDA_CPU_FIELD_LOCATION;
  } else if (strcmp(s, "gpu") == 0 || strcmp(s, "cuda") == 0) {
    ret = QUDA_CUDA_FIELD_LOCATION;
  } else {
    fprintf(stderr, "Error: invalid location %s\n", s);
    exit(1);
  }

  return ret;
}


QudaMemoryType
get_df_mem_type_ritz(const char* s)
{
  QudaMemoryType ret = QUDA_MEMORY_INVALID;

  if (strcmp(s, "device") == 0) {
    ret = QUDA_MEMORY_DEVICE;
  } else if (strcmp(s, "pinned") == 0) {
    ret = QUDA_MEMORY_PINNED;
  } else if (strcmp(s, "mapped") == 0) {
    ret = QUDA_MEMORY_MAPPED;
  } else {
    fprintf(stderr, "Error: invalid external library type %s\n", s);
    exit(1);
  }

  return ret;
}
