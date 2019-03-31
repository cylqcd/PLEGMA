#pragma once

/*
 * Functions for parsing and getting PLEGMA types, i.e. enums
 */

inline std::string get_file_format_str(FILE_WRITE_FORMAT format) {
  switch(format) {
  case ASCII_FORM:
    return "ascii";
  case HDF5_FORM:
    return "hdf5";
  case LIME_FORM:
    return "lime";
  default:
    PLEGMA_error("File format not found.\n");
    return "invalid";
  }
}

inline FILE_WRITE_FORMAT get_file_format(std::string s) {
  if(s=="ascii")
    return ASCII_FORM;
  else if(s=="hdf5")
    return HDF5_FORM;
  else if(s=="lime")
    return LIME_FORM;
  else {
    PLEGMA_error("invalid file format %s\n", s.c_str());
    return ASCII_FORM;
  }
}

inline std::string get_file_format_suffix(FILE_WRITE_FORMAT s) {
  std::string res;
  if(s==ASCII_FORM) res = ".dat";
  else if(s==HDF5_FORM) res =  ".h5";
  else if(s==LIME_FORM) res = "";
  else PLEGMA_error("invalid file format %s\n", get_file_format_str(s).c_str());
  return res;
}

inline std::string get_file_format_suffix(std::string s) {
  std::string res;
  if(s=="ascii") res = ".dat";
  else if(s=="hdf5") res =  ".h5";
  else if(s=="lime") res = "";
  else PLEGMA_error("invalid file format %s\n", s.c_str());
  return res;
}

inline std::string get_space_str(CORR_SPACE s) {
  switch(s) {
  case POSITION_SPACE:
    return "position";
  case MOMENTUM_SPACE:
    return "momentum";
  default:
    PLEGMA_error("Space not found.\n");
    return "invalid";
  }
}

inline CORR_SPACE get_space(std::string s) {
  if(s=="position")
    return POSITION_SPACE;
  else if(s=="momentum")
    return MOMENTUM_SPACE;
  else {
    PLEGMA_error("invalid space %s\n", s.c_str());
    return MOMENTUM_SPACE;
  }
}

inline WHICHPARTICLE get_particle(std::string s){
  WHICHPARTICLE par;
  if(s == "proton") par = PROTON;
  else if (s == "neutron")  par = NEUTRON;
  else PLEGMA_error("Particle %s is not implemented", s.c_str());
  return par;
}

inline WHICHPROJECTOR get_projector(std::string s){
  WHICHPROJECTOR proj;
  if(s == "P4_P") proj = P4_P;
  else if (s == "P4G5G1_P") proj = P4G5G1_P;
  else if (s == "P4G5G2_P") proj = P4G5G2_P;
  else if (s == "P4G5G3_P") proj = P4G5G3_P;
  else if (s == "P4_M") proj = P4_M;
  else if (s == "P4G5G1_M") proj = P4G5G1_M;
  else if (s == "P4G5G2_M") proj = P4G5G2_M;
  else if (s == "P4G5G3_M") proj = P4G5G3_M;
  else PLEGMA_error("Projector %s is not implemented", s.c_str());
  return proj;
}

inline std::string get_projector_str(WHICHPROJECTOR proj){
  std::string res="";
  if(proj==P4_P) res = "P4_P";
  else if(proj==P4G5G1_P) res = "P4G5G1_P";
  else if(proj==P4G5G2_P) res = "P4G5G2_P";
  else if(proj==P4G5G3_P) res = "P4G5G3_P";
  else if(proj==P4_M) res = "P4_M";
  else if(proj==P4G5G1_M) res = "P4G5G1_M";
  else if(proj==P4G5G2_M) res = "P4G5G2_M";
  else if(proj==P4G5G3_M) res = "P4G5G3_M";
  else PLEGMA_error("Projector not identified");
  return res;
}
