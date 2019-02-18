//===================//
// PLEGMA_enum       //
//===================//

enum COMPLEX{REAL,IMAG};
enum SOURCE_T{UNITY,RANDOM};
enum CORR_SPACE{POSITION_SPACE,MOMENTUM_SPACE};
enum FILE_WRITE_FORMAT{ASCII_FORM,HDF5_FORM};

enum ALLOCATION_FLAG{HOST,DEVICE,BOTH};

enum CLASS_ENUM{CUSTOM,SCALAR,SU3FIELD,GAUGE,VECTOR,PROPAGATOR,PROPAGATOR3D,VECTOR3D,QLOOPS};
enum GHOST_FLAG{NO_GHOSTS,FIRST_SIDE,FIRST_CORNER};
enum WHICHPARTICLE{PROTON,NEUTRON};
enum WHICHPROJECTOR{P4_P,P4G5G1_P,P4G5G2_P,P4G5G3_P,P4_M,P4G5G1_M,P4G5G2_M,P4G5G3_M}; // Do not change this order

enum THRP_TYPE{THRP_LOCAL2,THRP_NOETHER2,THRP_ONED2};

enum LATDIMS{DIM_X,DIM_Y,DIM_Z,DIM_T,N_DIMS};

enum GAMMAS {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43}; // Do not change this order
const std::string GAMMAS_STR[16] {"1","g1","g2","g3","g4","g5","g5g1","g5g2","g5g3","g5g4",
    "s12","s13","s23","s41","s42","s43"};
static inline std::string getGammasString(std::vector<GAMMAS> gammas) {
  std::string s = "";
  std::for_each(gammas.begin(), gammas.end(), [&] (GAMMAS n) {s += GAMMAS_STR[(int) n]+",";});
  return s;
}

enum ACCUM_TYPE{ACC_ZERO, ACC_PLUS, ACC_MINUS};
enum LEFTRIGHT {LEFT, RIGHT};
