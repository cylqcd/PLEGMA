#pragma once

/*
 * Functions for parsing and getting PLEGMA types, i.e. enums
 */

inline std::string get_file_format_str(FILE_FORMAT format) {
  switch(format) {
  case ASCII_FORMAT:
    return "ascii";
  case HDF5_FORMAT:
    return "hdf5";
  case LIME_FORMAT:
    return "lime";
  default:
    PLEGMA_error("File format not found.\n");
    return "invalid";
  }
}

inline FILE_FORMAT get_file_format(std::string s) {
  if(s=="ascii")
    return ASCII_FORMAT;
  else if(s=="hdf5")
    return HDF5_FORMAT;
  else if(s=="lime")
    return LIME_FORMAT;
  else {
    PLEGMA_error("invalid file format %s\n", s.c_str());
    return ASCII_FORMAT;
  }
}

inline std::string get_file_format_suffix(FILE_FORMAT s) {
  std::string res;
  if(s==ASCII_FORMAT) res = ".dat";
  else if(s==HDF5_FORMAT) res =  ".h5";
  else if(s==LIME_FORMAT) res = ".lime";
  else PLEGMA_error("invalid file format %s\n", get_file_format_str(s).c_str());
  return res;
}

inline std::string get_file_format_suffix(std::string s) {
  return get_file_format_suffix(get_file_format(s));
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

inline std::string get_particle_str(WHICHPARTICLE par) {
  switch(par) {
  case NEUTRON:
    return "neutron";
  case PROTON:
    return "proton";
  default:
    PLEGMA_error("Particle not found.\n");
    return "invalid";
  }
}

inline std::vector<GAMMAS> get_gammas(std::vector<std::string> s){
  std::vector<GAMMAS> g;
  for(size_t i=0;i<s.size();i++)
    {
      if(s[i] == "one") g.push_back(ONE); 
      else if (s[i] == "g1") g.push_back(G1);
      else if (s[i] == "g2") g.push_back(G2);
      else if (s[i] == "g3") g.push_back(G3);
      else if (s[i] == "g4") g.push_back(G4);
      else if (s[i] == "g5") g.push_back(G5);
      else if (s[i] == "g5g1") g.push_back(G5G1);
      else if (s[i] == "g5g2") g.push_back(G5G2);
      else if (s[i] == "g5g3") g.push_back(G5G3);
      else if (s[i] == "g5g4") g.push_back(G5G4);
      else if (s[i] == "s12") g.push_back(S12);
      else if (s[i] == "s13") g.push_back(S13);
      else if (s[i] == "s23") g.push_back(S23);
      else if (s[i] == "s41") g.push_back(S41);
      else if (s[i] == "s42") g.push_back(S42);
      else if (s[i] == "s43") g.push_back(S43);
      else PLEGMA_error("Gamma matrix %s is not implemented", s[i].c_str());
    }
  return g;
}

inline std::vector<std::string> get_gammas_str(std::vector<GAMMAS> g, std::vector<std::string> *s) {
  for(size_t i=0;i<g.size();i++)
    {
      if(g[i] == ONE) s->push_back("one"); 
      else if (g[i] == G1) s->push_back("g1");
      else if (g[i] == G2) s->push_back("g2");
      else if (g[i] == G3) s->push_back("g3");
      else if (g[i] == G4) s->push_back("g4");
      else if (g[i] == G5) s->push_back("g5");
      else if (g[i] == G5G1) s->push_back("g5g1");
      else if (g[i] == G5G2) s->push_back("g5g2");
      else if (g[i] == G5G3) s->push_back("g5g3");
      else if (g[i] == G5G4) s->push_back("g5g4");
      else if (g[i] == S12) s->push_back("s12");
      else if (g[i] == S13) s->push_back("s13");
      else if (g[i] == S23) s->push_back("s23");
      else if (g[i] == S41) s->push_back("s41");
      else if (g[i] == S42) s->push_back("s42");
      else if (g[i] == S43) s->push_back("s43");
      else s->push_back("invalid");
  }
  return *s;
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
  else if (s == "P_00") proj = P_00;
  else if (s == "P_01") proj = P_02;
  else if (s == "P_02") proj = P_02;
  else if (s == "P_03") proj = P_03;
  else if (s == "P_10") proj = P_10;
  else if (s == "P_11") proj = P_11;
  else if (s == "P_12") proj = P_12;
  else if (s == "P_13") proj = P_13;
  else if (s == "P_20") proj = P_20;
  else if (s == "P_21") proj = P_21;
  else if (s == "P_22") proj = P_22;
  else if (s == "P_23") proj = P_23;
  else if (s == "P_30") proj = P_30;
  else if (s == "P_31") proj = P_31;
  else if (s == "P_32") proj = P_32;
  else if (s == "P_33") proj = P_33;
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
  else if(proj==P_00) res + "P_00";
  else if(proj==P_01) res + "P_01";
  else if(proj==P_02) res + "P_02";
  else if(proj==P_03) res + "P_03";
  else if(proj==P_10) res + "P_10";
  else if(proj==P_11) res + "P_11";
  else if(proj==P_12) res + "P_12";
  else if(proj==P_13) res + "P_13";
  else if(proj==P_20) res + "P_20";
  else if(proj==P_21) res + "P_21";
  else if(proj==P_22) res + "P_22";
  else if(proj==P_23) res + "P_23";
  else if(proj==P_30) res + "P_30";
  else if(proj==P_31) res + "P_31";
  else if(proj==P_32) res + "P_32";
  else if(proj==P_33) res + "P_33";
  else PLEGMA_error("Projector not identified");
  return res;
}

inline WHICHFLAVOR get_flavor(std::string s){
  WHICHFLAVOR fl;
  if(s == "LIGHT") fl = LIGHT;
  else if (s == "STRANGE") fl = STRANGE;
  else if (s == "CHARM") fl = CHARM;
  else PLEGMA_error("Flavor %s is not implemented", s.c_str());
  return fl;
}

