//===================//
// PLEGMA_enum       //
//===================//

enum COMPLEX{REAL,IMAG};
enum SOURCE_T{UNITY,RANDOM};
enum CORR_SPACE{POSITION_SPACE,MOMENTUM_SPACE};
enum FILE_FORMAT{ASCII_FORMAT, HDF5_FORMAT, LIME_FORMAT, DEFAULT_FORMAT};

enum ALLOCATION_FLAG{EVERY=-1,NONE=0,HOST,DEVICE,BOTH};

enum CLASS_ENUM{CUSTOM,SCALAR,SU3FIELD,GAUGE,GAUGE3D,VECTOR,VECTOR3D,PROPAGATOR,PROPAGATOR3D,QLOOPS,FMUNU};
enum GHOST_FLAG{ALL_GHOSTS=-1,NO_GHOSTS,FIRST_SIDE,FIRST_CORNER,FIRST_VERTEX};
enum WHICHPARTICLE{PROTON,NEUTRON};
enum WHICHPROJECTOR{P4_P,P4G5G1_P,P4G5G2_P,P4G5G3_P,P4_M,P4G5G1_M,P4G5G2_M,P4G5G3_M, N_PROJS}; // Do not change this order and keep N_PROJS last
enum WHICHFLAVOR{LIGHT,STRANGE,CHARM};
enum THRP_TYPE{THRP_LOCAL2,THRP_NOETHER2,THRP_ONED2};

enum LATDIMS{DIM_X,DIM_Y,DIM_Z,DIM_T,N_DIMS};
enum ORIENTATION{DIR_PLUS, DIR_MINUS, DIR_BOTH};

enum GAMMAS {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43}; // Do not change this order
const std::string GAMMAS_STR[16] = {"1","g1","g2","g3","g4","g5","g5g1","g5g2","g5g3","g5g4",
    "s12","s13","s23","s41","s42","s43"};

enum GAMMAS_SCATT {ID,
                   G_1,
                   G_2,
                   G_3,
                   G_4,
                   G_5,
                   CG_1,
                   CG_2,
                   CG_3,
                   CG_1_G_4,
                   CG_2_G_4,
                   CG_3_G_4,
                   CG_1_G_4_G_5,
                   CG_2_G_4_G_5,
                   CG_3_G_4_G_5, 
                   G_5_G_1,
                   G_1_G_5,
                   G_5_G_2,
                   G_2_G_5,
                   G_5_G_3,
                   G_3_G_5,
                   G_5_G_4,
                   G_4_G_5,
                   G_5_CG_1,
                   CG_1_G_5,
                   G_5_CG_2,
                   CG_2_G_5,
                   G_5_CG_3,
                   CG_3_G_5,
                   C,
                   CG_5,
                   CG_4,
                   CG_5_G_4,
                   CG_4_G_5,
                   CG_5_G_4_G_5}; // Do not change this order
const std::string GAMMAS_SCATT_STR[35] = {"1",
                                          "g1",
                                          "g2",
                                          "g3",
                                          "g4",
                                          "g5",
                                          "cg1",
                                          "cg2",
                                          "cg3",
                                          "cg1g4",
                                          "cg2g4",
                                          "cg3g4",
                                          "cg1g4g5",
                                          "cg2g4g5",
                                          "cg3g4g5",
                                          "g5g1",
                                          "g1g5",
                                          "g5g2",
                                          "g2g5",
                                          "g5g3",
                                          "g3g5",
                                          "g5g4",
                                          "g4g5",
                                          "g5Cg1",
                                          "Cg1g5",
                                          "g5Cg2",
                                          "Cg2g5",
                                          "g5Cg3",
                                          "Cg3g5",
                                          "C",
                                          "Cg5",
                                          "Cg4",
                                          "Cg5g4",
                                          "Cg4g5",
                                          "Cg5g4g5"};
 

 
static inline std::string getGammasString(std::vector<GAMMAS> gammas) {
  std::string s = "";
  std::for_each(gammas.begin(), gammas.end(), [&] (GAMMAS n) {s += GAMMAS_STR[(int) n]+",";});
  return s;
}

enum ACCUM_TYPE{ACC_ZERO, ACC_PLUS, ACC_MINUS, ZERO_PLUS, ZERO_MINUS}; // ZERO_PLUS == ACC_ZERO
enum LEFTRIGHT {LEFT, RIGHT};
enum TOPO_CHARGE_DEF {PLAQUETTE,CLOVER,IMP_CLOVER};

enum ACTION {START, FINISH, DO_ALL};
