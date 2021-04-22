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


inline void readMomList() {
  std::ifstream file(pathListMomenta.c_str(), std::ifstream::in);
  if(file.fail()) PLEGMA_error("Cannot open file to read momentum list: %s\n",pathListMomenta.c_str());
  numMom=0;

  std::string line;
  while (std::getline(file, line))
    {
      std::istringstream ss(line);
      std::vector<float> new_vec;
      float v;
      while (ss >> v)      
	new_vec.push_back(v);
      listMomenta.push_back(new_vec);  
      numMom++;
	}
  file.close();
  PLEGMA_printf("\nList of read momenta:\n");
  for(int j=0; j<numMom; j++) 
      PLEGMA_printf("src[%d]: %.2f-%.2f-%.2f\n", j, listMomenta[j][0], listMomenta[j][1], listMomenta[j][2]);
      
}


