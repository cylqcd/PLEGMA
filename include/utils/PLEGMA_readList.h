#pragma once
/*
 * Functions that read from file the lists needed by PLEGMA_params.
 */

inline void readListStr(std::string listName, std::vector<std::string> &stdVec) {
  std::ifstream file(listName.c_str(),std::ifstream::in);
  if(file.fail()) PLEGMA_error("Cannot open file to read list of strings: %s\n",listName.c_str());
  std::string str;
  while(file >> str){
    stdVec.push_back(str);
  }
  file.close();
}

inline void readConfsList(){
  readListStr(pathListGaugeConfs,listGaugeConfs);
}

inline void readVecsList(){
  readListStr(pathListVecs,listVecs);
}

inline void readSourceList() {
  std::ifstream file(pathListSourcePositions.c_str(), std::ifstream::in);
  if(file.fail()) PLEGMA_error("Cannot open file to read source list: %s\n",pathListSourcePositions.c_str());
  int i=0;
  while (!file.eof() && i<numSourcePositions) {
    site source;
    file >> source;
    sourcePositions.push_back(source);
    i++;
  }
  file.close();
  if(i<numSourcePositions) {
    PLEGMA_warning("Read only %d source positions. Continuing with that ammount.\n", i);
      numSourcePositions = i;
  }
  if(verbosity) {
    PLEGMA_printf("\nList of read source positions:\n");
    for(int j=0; j<numSourcePositions; j++) {
      PLEGMA_printf("src[%d]: %d-%d-%d-%d\n", j, sourcePositions[j][0], sourcePositions[j][1], sourcePositions[j][2], sourcePositions[j][3]);
    }
  }
}

inline void readMomentaList() {
  std::ifstream file(pathListMomenta.c_str(), std::ifstream::in);
  if(file.fail()) PLEGMA_error("Cannot open file to read momenta list: %s\n", pathListMomenta.c_str());
  while (!file.eof()) {
    momentum mom;
    file >> mom;
    momenta.push_back(mom);
  }
  file.close();
  PLEGMA_printf("Read %d momenta.\n", momenta.size());
}

