#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
#include <algorithm>
#include <map>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>

#pragma once

#ifdef __GNUG__ // gnu C++ compiler
#include <cxxabi.h>
#include <stdlib.h>

#define _PRINT_(...){				\
  fprintf(stdout, __VA_ARGS__);			\
  fflush(stdout);				\
  }
#define _ERROR_(...){				\
  fprintf(stderr, __VA_ARGS__);			\
  fflush(stderr);				\
  exit(-1);					\
  }

   

static std::string demangle( const char* mangled_name ) {

    std::string result ;
    std::size_t len = 0 ;
    int status = 0 ;
    char* ptr = __cxxabiv1::__cxa_demangle( mangled_name, nullptr, &len, &status ) ;

    if( status == 0 ) result = ptr ; // hope that this won't throw
    else result = "demangle error" ;

    ::free(ptr) ;
    return result ;
}

#else
static std::string demangle( const char* name ) { return name ; }
#endif // _GNUG_

struct argument{std::string name, value;};

class Arguments{
protected:
  bool noOptions;
  bool isHelp;
  std::string prefixOpt;
  std::string forInput;
  std::vector<argument> args;
  std::string nameExec;

  void trimSpaceTab(std::string& str){
    if (str.empty()) return;
    int findht = str.find_first_of("#"); 
    str = str.substr(0,findht);
    int first = str.find_first_not_of(" \t");
    int last = str.find_last_not_of(" \t");
    str = str.substr(first, last-first+1);
  }

  void trimPrefix(std::string& str){
    if (str.empty()) return;
    checkPrefix(str);
    int jump = str.find(prefixOpt) + prefixOpt.size();
    str = str.substr(jump);
  }

  bool checkPrefix(std::string &str, bool throwExit = true){
    if(str.empty()){
      if(throwExit) _ERROR_("Error: Empty string got when checking for [%s] prefix\n",prefixOpt.c_str());
      return false;
    }
    size_t pos = str.find(prefixOpt);
    if(pos != 0 || (prefixOpt.size() == str.size()) ){
      if(throwExit) _ERROR_("Error: Option [%s] does not have the prefix [%s]\n",str.c_str(),prefixOpt.c_str());
      return false;
    }
    std::string sstr = str.substr(prefixOpt.size());
    if(sstr.find(prefixOpt) != std::string::npos){
      if(throwExit)_ERROR_("Error: Found dublications of [%s] in the option [%s]\n",prefixOpt.c_str(),str.c_str());
      return false;
    }
    return true;
  }

  void processInputFile(std::string infile){
    std::ifstream ifs(infile,std::ifstream::in);
    if(ifs.fail()) _ERROR_("Error: filename [%s] does not exist\n",infile.c_str());
    std::string line;
    while(std::getline(ifs,line)){
      argument arg;
      if(line.empty()) continue;
      trimSpaceTab(line);
      if(line.find(" ") == std::string::npos && line.find("\t") == std::string::npos)
	_ERROR_("Error: name [%s] does not have a value\n",line.c_str());
      int sp = line.find_first_of(" \t");
      arg.name = line.substr(0,sp);
      std::string rest = line.substr(sp);
      int bsp = rest.find_first_not_of(" \t");
      arg.value = rest.substr(bsp);
      args.push_back(arg);
    }
  }
public:
  Arguments(int argc, char *argv[]):noOptions(false),isHelp(false),prefixOpt("--"),forInput(prefixOpt+"inputFile"){
    std::vector<std::string> arg_cmd;
    nameExec = argv[0];
    for(int i = 1; i < argc; i++){
      std::string av(argv[i]);
      arg_cmd.push_back(av);
    }
    if(argc==1){ noOptions=true;}
    for(int i = 0; i < arg_cmd.size(); i++)
      if(arg_cmd[i] == prefixOpt+"help") isHelp=true;
    if(noOptions || isHelp) return;

    argument arg;
    for(int i = 0; i < arg_cmd.size(); i++){
      checkPrefix(arg_cmd[i]);
      if(arg_cmd[i] == forInput){
	processInputFile(arg_cmd.at(i+1));
	i++;
      }
      else{
	arg.name = arg_cmd[i];
	trimPrefix(arg.name);
	i++;
	if(i >= arg_cmd.size()) _ERROR_("Error: name [%s] does not have a value\n",arg.name.c_str());
	std::stringstream cs;
	while(!checkPrefix(arg_cmd.at(i),false)){
	  cs << " " << arg_cmd.at(i);
	  i++;
	  if(i >= arg_cmd.size()) break;
	}
	i--;
	std::string str = cs.str();
	trimSpaceTab(str);
	if(str.empty())_ERROR_("Error: name [%s] does not have a value\n",arg.name.c_str());
	arg.value = str;
	args.push_back(arg);
      }
    }
  }

  virtual ~Arguments(){}

  std::string get_value(std::string name) const{
    for(int i = 0 ; i < args.size(); i++)
      if(args[i].name == name)
	return args[i].value;
    return "";
  }
  std::vector<argument> get_args() const {return args;}

  void showArgsList() const{
    for(int i = 0 ; i < args.size(); i++)
      _PRINT_("%s %s\n",args[i].name.c_str(), args[i].value.c_str());
  }
    
  void areDuplications() const{
    for(int i = 0 ; i < args.size(); i++){
      std::string check_name = args[i].name;
      for(int j = 0 ; j < args.size(); j++)
	if(i!=j)
	  if(check_name == args[j].name)
	    _ERROR_("Error: Duplication of parameter [%s] found\n",check_name.c_str());
    }
  }
    
};

class Options : public Arguments{
private:
  std::string dressDesc;
  std::vector<std::string> descOpt; // to keep info about the description of the arguments
  std::vector<std::string> errorCollection; // to keep the errors for show at the end
  std::vector<std::string> listSetOpt; // list that keeps what is already set
  void usage(){
    _PRINT_("\n\n USAGE FOR %s\n",nameExec.c_str());
    int maxPos=0;
    for(int i = 0 ; i < descOpt.size(); i++){
      int pos = descOpt[i].find(dressDesc);
      if(pos > maxPos)maxPos=pos;
    }

    for(int i = 0 ; i < descOpt.size(); i++){
      std::replace(descOpt[i].begin(), descOpt[i].end(), '\t',' ');
      int pos = descOpt[i].find(dressDesc);
      std::string firstP = descOpt[i].substr(0,pos-1);
      std::string secondP = descOpt[i].substr(pos);
      std::string spaces(maxPos-pos+1,' ');
      descOpt[i] = firstP + spaces + secondP;
    }
    for(int i = 0 ; i < descOpt.size(); i++) _PRINT_("%s\n",descOpt[i].c_str());
  }
  template<typename T>
  void set(std::string name,std::stringstream &cs, T &v){
    bool check;
    if(typeid(T) == typeid(bool)){
      std::string tmp;
      check = static_cast<bool> ( cs >> tmp );
      if(!check){
	errorCollection.push_back("Error: Not enough arguments to unpack for option [" + name + "]");
	return;
      }
      if(tmp == "true")
	v = true;
      else if(tmp == "false")
	v = false;
      else
	errorCollection.push_back("Error: Boolean variable [" + name +"] accept either true or false ");
    }
    else{
      check = static_cast<bool>(cs >> v);
      if(!check) errorCollection.push_back("Error: Not enough arguments to unpack for option [" + name + "]");
    }
  }
    
  template<typename T, typename... Pars>
  void set(std::string name,std::stringstream &cs, T & p1, Pars & ... pars){
    set(name,cs,p1);
    set(name, cs, pars...);
  }

  void print(){;}
    
  template<typename T, typename... Pars>
  void print(T & p1, Pars & ... pars){
    std::stringstream cs;
    cs << " " << p1;
    _PRINT_("%s",cs.str().c_str());
    print(pars...);
  }

  template<typename T, typename... Pars>
  void print(std::string name, T & p1, Pars & ... pars){
    _PRINT_("%s",name.c_str());
    print(p1,pars...);
    _PRINT_("\n");
  }

  template<typename T>
  void print(std::string name, std::vector<T> &vec){
    _PRINT_("%s",name.c_str());
    std::stringstream cs;
    for(int i = 0 ; i < vec.size(); i++) cs << " " << vec[i];
    cs << std::endl;
    _PRINT_("%s\n",cs.str().c_str());
  }

  template<typename T1, typename T2>
  void print(std::string name, std::map<T1,T2> &tpl){
    _PRINT_("%s",name.c_str());
    std::stringstream cs;
    typename std::map<T1,T2>::iterator it_b = tpl.begin();
    while(it_b != tpl.end()){
      cs << " (" <<it_b->first << "," << it_b->second << ")";
      it_b++;
    }
    _PRINT_("%s\n",cs.str().c_str());
  }

  template<typename T, typename... Pars>
  std::string getOptTypes(T &p1, Pars & ... par){
    std::string res;
    res = "{" + demangle(typeid(p1).name()) + "} ";
    using expander = int[];
    (void)expander{0, (void(res += "{"  + demangle(typeid(std::forward<Pars>(par)).name()) + "} "), 0)...};
    return res;
  }
    
  template<typename... Pars>
  std::string getfullDesc(std::string name,std::string desc, Pars & ... par){
    return "[" + name + "] " + getOptTypes(par...) + dressDesc + " " + desc;
  }

  void checkIfSet(std::string name){
    for(int i = 0; i < listSetOpt.size(); i++)
      if(name == listSetOpt[i]) _ERROR_("Error: Option [%s] already set\n",name.c_str());
  }
public:
  Options(int argc,char **argv):Arguments(argc,argv),dressDesc("#++#"){;}
  ~Options(){checkErrors();}
    
  template<typename T, typename... Pars>
  bool set(std::string name, std::string desc, bool visualize, T &p1, Pars & ... par){
    std::string fullDesc = getfullDesc(name,desc,p1,par...);
    descOpt.push_back(fullDesc);
    if(noOptions || isHelp) return true;
    checkIfSet(name);
    std::stringstream cs;
    int countF=0;
    for(int i = 0 ; i < args.size(); i++)
      if(args[i].name == name){
	cs.clear();
	cs.str(args[i].value);
	set(name,cs,p1, par...);
	if(!cs.eof()) errorCollection.push_back("Error: More arguments to unpack than expected for option [" + name + "]");
	args.erase(args.begin()+i);
	countF++;
      }
    if(countF == 0) return false;
    else{
      if(visualize)print(name,p1,par...);
      listSetOpt.push_back(name);
      if(countF>1) _PRINT_("Warning: [%s] found %d times in the arguments. Last occurance is considered", name.c_str(), countF);
      return true;
    }
  }
    
  template<typename T, typename... Pars>
  void setForced(std::string name, std::string desc, bool visualize, T &p1, Pars & ... par){
    bool check = set(name,desc + " (FORCED)", visualize, p1, par...);
    if (!check) errorCollection.push_back("Error: [" + name + "] is not found in the arguments");
  }    

  
  template<typename T>
  bool set(std::string name, std::string desc, bool visualize, std::vector<T> &vec, int n=-1){
    std::string fullDesc = getfullDesc(name,desc,vec);
    descOpt.push_back(fullDesc);
    if(noOptions || isHelp) return true;
    checkIfSet(name);
    std::stringstream cs;
    int countF = 0;
    for(int i = 0 ; i < args.size(); i++)
      if(args[i].name == name){
	cs.clear();
	cs.str(args[i].value);
	int count=0;
	while(!cs.eof()){
	  vec.resize(count+1);
	  set(name,cs,vec[count]);
	  count++;
	}
	args.erase(args.begin()+i);
	countF++;
	if( (n>0) && (vec.size() != n) )
	  errorCollection.push_back("Error: " + name + " got wrong number of elements");
      }

    if(countF == 0) return false;
    else{
      if(visualize)print(name,vec);
      listSetOpt.push_back(name);
      if(countF>1) _PRINT_("Warning: [%s] found %d times in the arguments. Last occurance is considered", name.c_str(), countF);
      return true;
    }
  }

  template<typename T>
  void setForced(std::string name, std::string desc, bool visualize, std::vector<T> &vec, int n=-1){
    bool check = set(name,desc + " (FORCED)", visualize,vec,n);
    if (!check) errorCollection.push_back("Error: " + name + " is not found in the arguments");
  }

  template<typename T1, typename T2>
  bool set(std::string name, std::string desc, bool visualize, std::map<T1,T2> &tpl, int n=-1){
    std::string fullDesc = getfullDesc(name,desc,tpl);
    descOpt.push_back(fullDesc);
    if(noOptions || isHelp) return true;
    checkIfSet(name);
    std::stringstream cs;
    int countF=0;
    for(int i = 0 ; i < args.size(); i++)
      if(args[i].name == name){
	cs.clear();
	cs.str(args[i].value);
	while(!cs.eof()){
	  T1 t1;
	  T2 t2;
	  set(name,cs,t1);
	  set(name,cs,t2);
	  tpl.insert(std::make_pair(t1, t2));
	}
	args.erase(args.begin()+i);
	countF++;
	if( (n>0) && (tpl.size() != n) ){
	  errorCollection.push_back("Error: " + name + " got wrong number of elements");
	}
      }

    if(countF == 0) return false;
    else{
      if(visualize)print(name,tpl);
      listSetOpt.push_back(name);
      if(countF>1) _PRINT_("Warning: [%s] found %d times in the arguments. Last occurance is considered", name.c_str(), countF);      
      return true;
    }      
  }


  template<typename T1, typename T2>
  void setForced(std::string name, std::string desc, bool visualize, std::map<T1,T2> &tpl, int n=-1){
    bool check = set(name,desc + " (FORCED)",visualize,tpl,n);
    if (!check) errorCollection.push_back("Error: " + name + " is not found in the arguments");
  }

  void checkErrors(){
    if(isHelp){usage();exit(-1);}
    if(args.size() != 0){
      for(int i = 0; i<args.size(); i++)
	_PRINT_("Error: What option is %s\n", args[i].name.c_str() );
      usage();
      exit(-1);
    }
    if(errorCollection.size() <= 0) return;
    for(int i = 0 ; i < errorCollection.size(); i++) _PRINT_("%s\n",errorCollection[i].c_str());
    usage();
    exit(-1);
  }
};
