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
  ss <<	std::setprecision(16);
  ss <<	num;
  std::string s = ss.str();
  std::replace(s.begin(), s.end(), '.', 'p');
  std::string s2 = s.substr(0,s.find_last_not_of('0')+1);
  return s2;
}
