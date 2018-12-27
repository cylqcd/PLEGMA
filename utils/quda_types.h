#ifndef __MISC_H__
#define __MISC_H__

#include <quda.h>

QudaReconstructType get_recon(const char* s);
const char* get_recon_str(QudaReconstructType recon);

QudaPrecision   get_prec(const char* s);
const char* get_prec_str(QudaPrecision prec);

const char* get_gauge_order_str(QudaGaugeFieldOrder order);
const char* get_test_type(int t);
  const char* get_staggered_test_type(int t);
const char* get_unitarization_str(bool svd_only);

QudaMassNormalization get_mass_normalization_type(const char* s);
const char* get_mass_normalization_str(QudaMassNormalization);

QudaVerbosity get_verbosity_type(const char* s);
const char* get_verbosity_str(QudaVerbosity);

QudaMatPCType get_matpc_type(const char* s);
const char* get_matpc_str(QudaMatPCType);

QudaSolveType get_solve_type(const char* s);
const char* get_solve_str(QudaSolveType);

QudaSchwarzType get_schwarz_type(const char* s);

QudaTwistFlavorType get_flavor_type(const char* s);

int get_rank_order(const char* s);

QudaDslashType get_dslash_type(const char* s);
const char* get_dslash_str(QudaDslashType type);

QudaInverterType get_solver_type(const char* s);
const char* get_solver_str(QudaInverterType type);

const char* get_quda_ver_str();

QudaExtLibType get_solve_ext_lib_type(const char* s);

QudaFieldLocation get_location(const char* s);

QudaMemoryType get_df_mem_type_ritz(const char* s);

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3
#define TDOWN 4
#define ZDOWN 5
#define YDOWN 6
#define XDOWN 7
#define OPP_DIR(dir)    (7-(dir))
#define GOES_FORWARDS(dir) (dir<=3)
#define GOES_BACKWARDS(dir) (dir>3)


#endif


