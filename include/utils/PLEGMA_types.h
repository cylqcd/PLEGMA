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
    PLEGMA_error("invalid file format %s\n", s);
    return ASCII_FORM;
  }
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
    PLEGMA_error("invalid space %s\n", s);
    return MOMENTUM_SPACE;
  }
}
