#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
#include <type_traits>

#pragma once

template<typename T>
std::string convNumToStr(T num){
  if(!std::is_floating_point<T>::value) PLEGMA_error("Only floating point types are allowed");
  std::stringstream ss;
  ss <<	std::fixed;
  ss <<	std::setprecision(12);
  ss <<	num;
  std::string s = ss.str();
  std::replace(s.begin(), s.end(), '.', 'p');
  std::string s2 = s.substr(0,s.find_last_not_of('0')+1);
  return s2;
}

template<typename T>
std::vector<int> clearDuplicates(std::vector<T> &vec){
  if(std::is_pointer<T>::value) PLEGMA_error("Do not know how to remove duplicates from pointer containers");
  std::sort(vec.begin(), vec.end());
  std::map<T,int> counter;
  std::vector<int> res;
  for(auto const &f: vec) counter[f]++;
  vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
  for(auto const &b: counter) res.push_back(b.second);
  return res;
}

template<typename T1>
std::string write_std_vecs_unpacker(int i, const std::vector<T1> vec){
  std::stringstream ss;
  if(i>= vec.size())  ss << "NA" << std::endl;
  else   ss << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << vec[i] << std::endl;
  return ss.str();
}

template<typename T1, typename ...T2>
std::string write_std_vecs_unpacker(int i, const std::vector<T1>& vec,const std::vector<T2>& ...vecs){
  std::stringstream ss;
  if(i>= vec.size())  ss << "NA" << "\t" << write_std_vecs_unpacker(i,vecs...);
  else ss << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << vec[i] << "\t" << write_std_vecs_unpacker(i,vecs...);
  return ss.str();
}

template<typename T1, typename ...T2>
void write_std_vecs(std::string filename,bool isAppend,const std::vector<T1>& vec,const std::vector<T2>& ...vecs){
  std::ofstream file;
  if(isAppend) file.open(filename,std::ofstream::out | std::ofstream::app);
  else file.open(filename);
  if(!file) PLEGMA_error("Cannot open file %s\n", filename.c_str());
  for(int i = 0; i < vec.size(); i++)
    file << write_std_vecs_unpacker(i,vec,vecs...);
}

inline std::string getDateAndTime(){
  auto time = std::time(nullptr);
  std::stringstream ss;
  ss << std::put_time(std::gmtime(&time), "%F %T%z");
  return "Date: " + ss.str();
}

inline std::string basename(std::string str,char c){
  size_t found = str.find_last_of(c);
  if(found == std::string::npos) PLEGMA_error("Cannot get the baseanem of string=%s with delimiter=%c\n",str.c_str(),c);
  std::string suff = str.substr(found+1,std::string::npos);
  return suff;
}

inline void cleanFile(std::string filename){
  std::ofstream file(filename);
  if(file.fail()) PLEGMA_error("Cannot open file to clean it: %s\n",filename.c_str());
  file.close();
}
