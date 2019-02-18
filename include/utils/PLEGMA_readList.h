#pragma once

/*
 * Functions that read from file the lists needed by PLEGMA_params.
 */

inline void readSourceList() {
  hostMalloc(sourcePositions, N_DIMS*numSourcePositions*sizeof(int));
  std::ifstream file(pathListSourcePositions.c_str(), std::ifstream::in);
  int i=0;
  while (!file.eof() || i<N_DIMS*numSourcePositions) {
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

inline void readTSinkList() {
  hostMalloc(tSink, numTSink*sizeof(tSink));
  std::ifstream file(pathListTSink.c_str(), std::ifstream::in);
  int i=0;
  while (!file.eof() || i<numTSink) {
    file >> tSink[i];
    i++;
  }
  file.close();
  if(i<numTSink) {
    PLEGMA_warning("Read only %d T sinks. Continuing with that ammount.\n", i);
    numTSink = i;
  }
  if(verbosity) {
    PLEGMA_printf("\nList of read T sink:\n");
    for(int j=0; j<numTSink; j++) {
	PLEGMA_printf("tsink[%d]: %d\n", j, tSink[j]);
    }
  }
}

inline void readProjList() {
  hostMalloc(proj, numProj*sizeof(proj));
  std::ifstream file(pathListProj.c_str(), std::ifstream::in);
  int i=0;
  std::string tmpString;
  std::string proj_str[]={"P4_P","P4G5G1_P","P4G5G3_P","P4_M","P4G5G1_M","P4G5G2_M","P4G5G3_M"};
  while (!file.eof() || i<numProj) {
    file >> tmpString;
    std::string *p = std::find (proj_str, proj_str+(int)N_PROJS, tmpString);
    proj[i] = (WHICHPROJECTOR) (int) (p - proj_str);
    if (proj[i] == N_PROJS) PLEGMA_error("Proj %s not found. Options are P4_P, P4G5G1_P, P4G5G2_P, P4G5G3_P, P4_M, P4G5G1_M, P4G5G2_M, P4G5G3_M.\n");
    i++;
  }
  file.close();
  if(i<numProj) {
    PLEGMA_warning("Read only %d projectors. Continuing with that ammount.\n", i);
    numProj = i;
  }
  if(verbosity) {
    PLEGMA_printf("\nList of read projectors:\n");
    for(int j=0; j<numProj; j++) {
      PLEGMA_printf("proj[%d]: %d\n", j, proj_str[(int)proj[j]]);
    }
  }
}
