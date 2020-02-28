#pragma once
#include "PLEGMA_templates.h"
struct argument{std::string name, value;};

class Arguments{
protected:
  bool noOptions;
  bool isHelp;
  std::string prefixOpt;
  std::string forInput;
  std::vector<argument> args;
  std::string nameExec;
  std::vector<std::string> errorCollection; // to keep the errors for show at the end

  void trimSpaceTab(std::string& str){
    if (str.empty()) return;
    int findht = str.find_first_of("#"); 
    str = str.substr(0,findht);
    int first = str.find_first_not_of(" \t");
    int last = str.find_last_not_of(" \t");
    str = str.substr(first, last-first+1);
  }

  bool isCommented(std::string& str){
    if (str.empty()) return false;
    if(str.find_first_of("#") == 0) return true;
    else return false;
  }
  
  void trimPrefix(std::string& str){
    if (str.empty()) return;
    checkPrefix(str);
    int jump = str.find(prefixOpt) + prefixOpt.size();
    str = str.substr(jump);
  }

  bool checkPrefix(std::string &str, bool throwExit = true){
    if(str.empty()){
      if(throwExit) errorCollection.push_back("ERROR: Empty string got when checking for ["+prefixOpt+"] prefix\n");
      return false;
    }
    size_t pos = str.find(prefixOpt);
    if(pos != 0 || (prefixOpt.size() == str.size()) ){
      if(throwExit) errorCollection.push_back("ERROR: Option ["+str+"] does not have the prefix ["+prefixOpt.c_str()+"]\n");
      return false;
    }
    std::string sstr = str.substr(prefixOpt.size());
    if(sstr.find(prefixOpt) != std::string::npos){
      if(throwExit) errorCollection.push_back("ERROR: Found dublications of ["+prefixOpt+"] in the option ["+str+"]\n");
      return false;
    }
    return true;
  }

  void processInputFile(std::string infile){
    std::ifstream ifs(infile,std::ifstream::in);
    if(ifs.fail())  {
      errorCollection.push_back("Error: filename ["+infile+"] does not exist\n");
      return;
    }
    std::string line;
    while(std::getline(ifs,line)){
      argument arg;
      if(line.empty()) continue;
      if(isCommented(line)) continue;
      trimSpaceTab(line);
      if(line.find(" ") == std::string::npos && line.find("\t") == std::string::npos)
	errorCollection.push_back("Error: name ["+line+"] does not have a value\n");
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
    for(size_t i = 0; i < arg_cmd.size(); i++)
      if(arg_cmd[i] == prefixOpt+"help") isHelp=true;
    if(noOptions || isHelp) return;

    argument arg;
    for(size_t i = 0; i < arg_cmd.size(); i++){
      checkPrefix(arg_cmd[i]);
      if(arg_cmd[i] == forInput){
	processInputFile(arg_cmd.at(i+1));
	i++;
      }
      else{
	arg.name = arg_cmd[i];
	trimPrefix(arg.name);
	i++;
	if(i >= arg_cmd.size()) errorCollection.push_back("Error: name ["+arg.name+"] does not have a value\n");
	std::stringstream cs;
	while(!checkPrefix(arg_cmd.at(i),false)){
	  cs << " " << arg_cmd.at(i);
	  i++;
	  if(i >= arg_cmd.size()) break;
	}
	i--;
	std::string str = cs.str();
	trimSpaceTab(str);
	if(str.empty()) errorCollection.push_back("Error: name ["+arg.name+"] does not have a value\n");
	arg.value = str;
	args.push_back(arg);
      }
    }
  }

  virtual ~Arguments(){}

  bool getIsHelp() const{return isHelp;}
  
  std::string get_value(std::string name) const{
    for(size_t i = 0 ; i < args.size(); i++)
      if(args[i].name == name)
	return args[i].value;
    return "";
  }
  std::vector<argument> get_args() const {return args;}

  void showArgsList() const{
    for(size_t i = 0 ; i < args.size(); i++)
      PLEGMA_printf("%s %s\n",args[i].name.c_str(), args[i].value.c_str());
  }
    
  void areDuplications() {
    for(size_t i = 0 ; i < args.size(); i++){
      std::string check_name = args[i].name;
      for(size_t j = 0 ; j < args.size(); j++)
	if(i!=j)
	  if(check_name == args[j].name)
	    errorCollection.push_back("Error: Duplication of parameter ["+check_name+"] found\n");
    }
  }
    
};

class Options : public Arguments{
private:
  std::string dressDesc;
  std::vector<std::string> descOpt; // to keep info about the description of the arguments
  std::vector<std::string> listSetOpt; // list that keeps what is already set
  bool isOpen;
  void usage(){
    PLEGMA_printf("\n\n USAGE FOR %s\n",nameExec.c_str());
    int maxPos=0;
    for(size_t i = 0 ; i < descOpt.size(); i++){
      int pos = descOpt[i].find(dressDesc);
      if(pos > maxPos)maxPos=pos;
    }

    for(size_t i = 0 ; i < descOpt.size(); i++){
      std::replace(descOpt[i].begin(), descOpt[i].end(), '\t',' ');
      int pos = descOpt[i].find(dressDesc);
      std::string firstP = descOpt[i].substr(0,pos-1);
      std::string secondP = descOpt[i].substr(pos);
      std::string spaces(maxPos-pos+1,' ');
      descOpt[i] = firstP + spaces + secondP;
    }
    for(size_t i = 0 ; i < descOpt.size(); i++) PLEGMA_printf("%s\n",descOpt[i].c_str());
  }
  template<typename T>
  void set(std::string name,std::stringstream &cs, T &v){
    bool check;
    check = static_cast<bool>(cs >> v);
    if(!check) errorCollection.push_back("Error: Not enough arguments to unpack for option [" + name + "]");
  }
  void set(std::string name,std::stringstream &cs, bool &v){
    bool check;
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
    
  template<typename T, typename... Pars>
  void set(std::string name,std::stringstream &cs, T & p1, Pars & ... pars){
    set(name,cs,p1);
    set(name, cs, pars...);
  }

  template<typename... Pars>
  void print(Pars & ... pars){
    PLEGMA_printf("%s",toString(pars...).c_str());
  }

  template<typename... Pars>
  void print(std::string name, Pars & ... pars){
    PLEGMA_printf("%s\n",(name+toString(pars...)).c_str());
  }

  
  template<typename T, typename... Pars>
  std::string getOptTypes(T &p1, Pars & ... par){
    std::string res;
    res = "{" + demangle(typeid(p1).name()) + "} ";
    using expander = int[];
    (void)expander{0, (void(res += "{"  + demangle(typeid(std::forward<Pars>(par)).name()) + "} "), 0)...};
    return res;
  }

  template<typename T>
  std::string getOptTypes(std::vector<T> &vec){
    std::string res;
    res = "{std::vector";
    res += "<" + demangle(typeid(T).name()) + ">} ";
    return res;
  }

  template<typename T1, typename T2>
  std::string getOptTypes(std::map<T1,T2> &tpl){
    std::string res;
    res = "{std::map";
    res += "<" + demangle(typeid(T1).name()) + "," + demangle(typeid(T2).name()) + ">} ";
    return res;
  }


  template<typename... Pars>
  std::string getfullDesc(std::string name,std::string desc, Pars & ... par){
    return "[" + name + "] " + getOptTypes(par...) + dressDesc + " " + desc + ", default:" + toString(par...);
  }

  void checkIfSet(std::string name){
    for(size_t i = 0; i < listSetOpt.size(); i++)
      if(name == listSetOpt[i]) errorCollection.push_back("Error: Option ["+name+"] already set\n");
  }
public:
  Options(int argc,char **argv):Arguments(argc,argv),dressDesc("#++#"),isOpen(true){
  }
  ~Options(){close();}
    
  template<typename T, typename... Pars>
  bool set(std::string name, std::string desc, int visualize, T &p1, Pars & ... par){
    if(!isOpen){PLEGMA_error("Options are closed you cannot set");}
    std::string fullDesc = getfullDesc(name,desc,p1,par...);
    descOpt.push_back(fullDesc);
    if(noOptions) return false;
    checkIfSet(name);
    std::stringstream cs;
    int countF=0;
    for(size_t i = 0 ; i < args.size(); i++)
      if(args[i].name == name){
	cs.clear();
	cs.str(args[i].value);
	set(name,cs,p1, par...);
	if(!cs.eof()) errorCollection.push_back("Error: More arguments to unpack than expected for option [" + name + "]");
	args.erase(args.begin()+i);
	countF++;
      }
    if(countF == 0) { if(visualize>1) print(name,p1,par...); return false; }
    else{
      if(visualize)print(name,p1,par...);
      listSetOpt.push_back(name);
      if(countF>1) PLEGMA_printf("Warning: [%s] found %d times in the arguments. Last occurance is considered", name.c_str(), countF);
      return true;
    }
  }
    
  template<typename T, typename... Pars>
  void setForced(std::string name, std::string desc, int visualize, T &p1, Pars & ... par){
    if(!isOpen){PLEGMA_error("Options are closed you cannot set");}
    bool check = set(name,desc + " (FORCED)", visualize, p1, par...);
    if (!check && !getIsHelp()) errorCollection.push_back("Error: [" + name + "] is not found in the arguments");
  }    

  
  template<typename T>
  bool set(std::string name, std::string desc, int visualize, std::vector<T> &vec, int n=-1){
    if(!isOpen){PLEGMA_error("Options are closed you cannot set");}
    std::string fullDesc = getfullDesc(name,desc,vec);
    descOpt.push_back(fullDesc);
    if(noOptions) return false;
    checkIfSet(name);
    std::stringstream cs;
    int countF = 0;
    for(size_t i = 0 ; i < args.size(); i++)
      if(args[i].name == name){
	cs.clear();	
	vec.clear();
	// We use "-" just to empty a vector
	if(args[i].value != "-") {
	  cs.str(args[i].value);
	  int count=0;
	  while(!cs.eof()){
	    vec.resize(count+1);
	    set(name,cs,vec[count]);
	    count++;
	  }
	}
	args.erase(args.begin()+i);
	countF++;
	if( (n>0) && ((int) vec.size() != n) )
	  errorCollection.push_back("Error: " + name + " got wrong number of elements");
      }

    if(countF == 0) { if(visualize>1) print(name,vec); return false; }
    else{
      if(visualize)print(name,vec);
      listSetOpt.push_back(name);
      if(countF>1) PLEGMA_printf("Warning: [%s] found %d times in the arguments. Last occurance is considered", name.c_str(), countF);
      return true;
    }
  }

  template<typename T>
  void setForced(std::string name, std::string desc, int visualize, std::vector<T> &vec, int n=-1){
    if(!isOpen){PLEGMA_error("Options are closed you cannot set");}
    bool check = set(name,desc + " (FORCED)", visualize,vec,n);
    if (!check && !getIsHelp()) errorCollection.push_back("Error: " + name + " is not found in the arguments");
  }

  template<typename T1, typename T2>
  bool set(std::string name, std::string desc, int visualize, std::map<T1,T2> &tpl, int n=-1){
    if(!isOpen){PLEGMA_error("Options are closed you cannot set");}
    std::string fullDesc = getfullDesc(name,desc,tpl);
    descOpt.push_back(fullDesc);
    if(noOptions) return false;
    checkIfSet(name);
    std::stringstream cs;
    int countF=0;
    for(size_t i = 0 ; i < args.size(); i++)
      if(args[i].name == name){
	cs.clear();
	cs.str(args[i].value);
	while(!cs.eof()){
	  T1 t1;
	  T2 t2;
	  set(name,cs,t1);
	  set(name,cs,t2);
	  auto search = tpl.find(2);
	  if (search != tpl.end()) {
	    tpl[t1] = t2;
	  } else {
	    tpl.insert(std::make_pair(t1, t2));
	  }
	}
	args.erase(args.begin()+i);
	countF++;
	if( (n>0) && ((int) tpl.size() != n) ){
	  errorCollection.push_back("Error: " + name + " got wrong number of elements");
	}
      }

    if(countF == 0) { if(visualize>1) print(name,tpl); return false; }
    else{
      if(visualize)print(name,tpl);
      listSetOpt.push_back(name);
      if(countF>1) PLEGMA_printf("Warning: [%s] found %d times in the arguments. Last occurance is considered", name.c_str(), countF);      
      return true;
    }      
  }


  template<typename T1, typename T2>
  void setForced(std::string name, std::string desc, int visualize, std::map<T1,T2> &tpl, int n=-1){
    if(!isOpen){PLEGMA_error("Options are closed you cannot set");}
    bool check = set(name,desc + " (FORCED)",visualize,tpl,n);
    if (!check && !getIsHelp()) errorCollection.push_back("Error: " + name + " is not found in the arguments");
  }

  void checkErrors(){
    if(errorCollection.size() <= 0) return;
    for(size_t i = 0 ; i < errorCollection.size(); i++) PLEGMA_printf("%s\n",errorCollection[i].c_str());
    PLEGMA_exit(-1);
  }

  void close(){
    if(isHelp){usage(); PLEGMA_exit(-1);}
    checkErrors();
    if(args.size() != 0){
      for(size_t i = 0; i<args.size(); i++)
	PLEGMA_printf("Error: What option is %s\n", args[i].name.c_str() );
      usage();
      PLEGMA_exit(-1);
    }
    isOpen=false;
  }
};
