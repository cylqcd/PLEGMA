#pragma once
/*
 * Functions that read from file the lists needed by PLEGMA_params.
 */

inline void readConfsList() {
  std::ifstream file(pathListGaugeConfs.c_str(),std::ifstream::in);
  if(file.fail()) PLEGMA_error("Cannot open file to read confs list: %s\n",pathListGaugeConfs.c_str());
  std::string str;
  while(file >> str){
    listGaugeConfs.push_back(str);
  }
  file.close();
}

inline void readSourceList() {
  hostMalloc(sourcePositions, N_DIMS*numSourcePositions*sizeof(int));
  std::ifstream file(pathListSourcePositions.c_str(), std::ifstream::in);
  if(file.fail()) PLEGMA_error("Cannot open file to read source list: %s\n",pathListSourcePositions.c_str());
  int i=0;
  while (!file.eof() && i<N_DIMS*numSourcePositions) {
    file >> sourcePositions[i/N_DIMS][i%N_DIMS];
    i++;
  }
  file.close();
  if(i<N_DIMS*numSourcePositions) {
    PLEGMA_warning("Read only %d source positions. Continuing with that ammount.\n", i/N_DIMS);
      numSourcePositions = i/N_DIMS;
  }
  if(verbosity) {
    PLEGMA_printf("\nList of read source positions:\n");
    for(int j=0; j<numSourcePositions; j++) {
      PLEGMA_printf("src[%d]: %d-%d-%d-%d\n", j,  sourcePositions[j][0], sourcePositions[j][1], sourcePositions[j][2], sourcePositions[j][3]);
    }
  }
}

