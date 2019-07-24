#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <list>

using namespace plegma;
using namespace quda;

template <typename T>
struct InVar{

  T* variable; 
  std::string name;
  std::vector<T> values;
  bool continuous;

  void add_value(T value) {
    auto it = std::find(values.begin(), values.end(), value);
    if (it >= values.end()) {
      if(!continuous) {
	values.insert(values.begin(),value);
      } else {
	auto it = values.begin();
	while(*it < value && it<values.end()){it++;}
	values.insert(it, value);
      }
    }
  }

  T get() {
    return *variable;
  }
  void set(T value) {
    if(get() != value) {
      add_value(value);
      *variable = value;
      if(HGC_verbosity>1) PLEGMA_printf(("MG_Tuner: Set "+name+" to "+std::to_string(value)+"\n").c_str());
    }
  }

  std::string get_info() {
    std::string info = "Variable: "+name+", current value: "+std::to_string(get())+", values to test: ";
    for(auto v: values)
      info+=std::to_string(v)+", ";
    info += "\n";
    return info;      
  }

  InVar( T* variable, std::string name, std::vector<T> values, bool continuous = false): variable(variable), name(name), values(values), continuous(continuous){
    add_value(*variable); // adding current value
    if(HGC_verbosity>1) PLEGMA_printf(get_info().c_str());
  }

};

//Useful for unpacking tuple
//Taken from https://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer/7858971#7858971
template<int ...> struct seq {};

template<int N, int ...S> struct gens : gens<N-1, N-1, S...> {};

template<int ...S> struct gens<0, S...>{ typedef seq<S...> type; };

template<class ...types>
struct SolverTimings{

  QUDA_solver &solver;
  PLEGMA_Vector<double> &vectorInOut;
  std::tuple<InVar<types>*...> variables;
  std::vector<std::tuple<double,types...>> timings;

  SolverTimings(QUDA_solver &solver, PLEGMA_Vector<double> &vectorInOut, InVar<types>*... var) : solver(solver), vectorInOut(vectorInOut) {
    this->variables = std::make_tuple(var...);
  }

  template<int...S>
  void _append(double time,seq<S...>){
    timings.push_back(std::make_tuple(time,std::get<S>(variables)->get()...));
  }
  void append(double time){
    _append(time,typename gens<sizeof...(types)>::type());
  }
  
  template<std::size_t I = 0, typename time_t>
  inline typename std::enable_if<I == sizeof...(types), void>::type
  remove_not_mathing(time_t &) {
  }
  template<std::size_t I = 0, typename time_t>
  inline typename std::enable_if<I < sizeof...(types), void>::type
  remove_not_mathing(time_t &time) {
    if(time.empty()) return;
    auto value = std::get<I>(variables)->get();
    for (auto it = time.begin(); it < time.end(); ) {
      auto it_value = std::get<I+1>(*it);
      if(it_value != value)
	it = time.erase(it);
      else
	it++;
    }
    remove_not_mathing<I+1>(time);
  }

  //Method for printing the tuple elements
  template<std::size_t I = 0, typename ...Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  print(std::tuple<Tp...>& t) {
  }
  template<std::size_t I = 0, typename ...Tp>
  inline typename std::enable_if< (I > 0 & I < sizeof...(Tp)), void>::type
  print(std::tuple<Tp...>& t) {
    std::string name = (std::get<I-1>(variables)->name);
    PLEGMA_printf((name + ": " + std::to_string(std::get<I>(t))+", ").c_str());
    print<I + 1, Tp...>(t);
  }
  template<std::size_t I = 0, typename ...Tp>
  inline typename std::enable_if<I == 0, void>::type
  print(std::tuple<Tp...>& t) {
    std::string name = "time: ";
    PLEGMA_printf((name + std::to_string(std::get<I>(t))+", ").c_str());
    print<I + 1, Tp...>(t);
  }
  template<std::size_t I = 0>
  inline typename std::enable_if<I == sizeof...(types), void>::type
  printVars() {
  }
  template<std::size_t I = 0>
  inline typename std::enable_if< I < sizeof...(types), void>::type
  printVars() {
    std::string name = (std::get<I>(variables)->name);
    PLEGMA_printf((name + ": " + std::to_string(std::get<I>(variables)->get())+", ").c_str());
    printVars<I + 1>();
  }
    
  double apply(){
    auto time_copy = timings;
    remove_not_mathing(time_copy);
    if(time_copy.empty()) {
      double t0, t1;
      int sources[4]={0};
      sources[1]=1;
      solver.UpdateSolver();
      // Doing one iter for performing tuning where needed
      vectorInOut.pointSource(sources, 0, 0, DEVICE);
      solver.runOneIter(vectorInOut, vectorInOut);
      vectorInOut.pointSource(sources, 0, 0, DEVICE);
      t1=MPI_Wtime();
      solver.solve(vectorInOut, vectorInOut);
      t0 = MPI_Wtime()-t1;
      MPI_Allreduce(&t0, &t1, 1, MPI_Type(t0), MPI_MAX, MPI_COMM_WORLD);
      // Rescaling the time with the residual
      t1=t1*log(tol)/log(solver.getSolverParam()->true_res);
      append(t1);
      PLEGMA_printf("MG_Tuner: new ");
      print(timings[timings.size()-1]);
      PLEGMA_printf("\n");
      return t1;
    } else {
      return std::get<0>(time_copy[0]);
    }
  }

  void printBest(){
    int best_index = 0;
    double best_time = std::get<0>(timings[0]);
    for(size_t i=1; i<timings.size(); i++) {
      if(best_time > std::get<0>(timings[i])) {
	best_index = i;
	best_time = std::get<0>(timings[i]);
      }
    }
    PLEGMA_printf("MG_Tuner: set of best parameters \n MG_Tuner: ");
    print(timings[best_index]);
    PLEGMA_printf("\n");
  }
};
  
template<class T, class ...types>
struct Minimizer{

  T* call;
  std::tuple<InVar<types>*...> variables;
  bool enabled;
  
  Minimizer(T* call, bool enabled, InVar<types>*... var) : call(call), enabled(enabled) {
    this->variables = std::make_tuple(var...);
  }

  template<class T1>
  double call_back(T1* c){
    return c->apply();
  }
  template<std::size_t I = 0, class... Args>
  inline typename std::enable_if<I == sizeof...(Args), void>::type
  _call_back(std::tuple<Args...>* c, double& time ) {
  }
  template<std::size_t I = 0, class... Args>
  inline typename std::enable_if<I < sizeof...(Args), void>::type
  _call_back(std::tuple<Args...>* c, double& time )  {
    time = std::get<I>(*c)->apply();
    _call_back<I+1, Args...>(c, time);
  }
  template<class... Args>
  double call_back(std::tuple<Args...>* c){
    double time;
    _call_back(c, time);
    return time;
  }

  template<std::size_t I = 0, typename disc_t>
  inline typename std::enable_if<I == sizeof...(types), void>::type
  select_discrete(disc_t& ) {
  }
  template<std::size_t I = 0, typename disc_t>
  inline typename std::enable_if<I < sizeof...(types), void>::type
  select_discrete(disc_t& discrete) {
    if (std::get<I>(variables)->continuous == false) discrete.push_back({I, std::get<I>(variables)->values.size()});
    select_discrete<I + 1>(discrete);
  }
  
  template<std::size_t I = 0, typename disc_t>
  inline typename std::enable_if<I == sizeof...(types), void>::type
  tuple_iterator(disc_t, size_t) {
  }
  template<std::size_t I = 0, typename disc_t>
  inline typename std::enable_if<I < sizeof...(types), void>::type
  tuple_iterator(disc_t discr, size_t i) {
    if(I == discr[0][0]) {
      auto elem=std::get<I>(variables);
      elem->set(elem->values[i%discr[0][1]]);
      i/=discr[0][1];
      discr.erase(discr.begin());
    }
    tuple_iterator<I+1>(discr,i);
  }

  template<std::size_t I = 0>
  inline typename std::enable_if<I == sizeof...(types), void>::type
  find_min(bool&, double &best_time) {
  }
  template<std::size_t I = 0>
  inline typename std::enable_if<I < sizeof...(types), void>::type
  find_min(bool &changed, double &best_time) {
    auto elem = std::get<I>(variables);
    if(elem->continuous == true) {
      auto it = std::find(elem->values.begin(), elem->values.end(), elem->get());
      if (it >= elem->values.end()) {
	it = elem->values.begin();
	elem->set(*it);
      }
      
      double t0 = call_back(call);
      
      bool do_it=true;
      bool Ichanged=false;
      while (do_it && it+1 < elem->values.end()) {
	it++;
	elem->set(*it);
	double tp = call_back(call);
	if(tp<t0) {
	  Ichanged=true;
	  changed=true;
	  t0=tp;
	} else {
	  do_it=false;
	  it--;
	  elem->set(*it);
	}
      }
      if(!Ichanged) {
	do_it=true;
	while (do_it && it-1 >= elem->values.begin()) {
	  it--;
	  elem->set(*it);
	  double tm = call_back(call);
	  if(tm<t0) {
	    Ichanged=true;
	    changed=true;
	    t0=tm;
	  } else {
	    do_it=false;
	    it++;
	    elem->set(*it);
	  }
	}
      }
      if(best_time < 0 || t0 < best_time)
	best_time = t0;
    }

    find_min<I+1>(changed, best_time);
  }


  double apply(){
    if(enabled) {
      std::vector<std::array<size_t,2>> discrete;
      select_discrete(discrete);
    
      int nCombinations = 1;
      for(auto a : discrete) nCombinations*=a[1];

      int best_index = 0;
      int best_time = -1;
      // running over all the possible combinations of discrete parameters
      for(int i=0; i < nCombinations; i++){
	if(!discrete.empty()) {
	  if(HGC_verbosity>1) PLEGMA_printf("MG_Tuner: iteration on discrete values %d/%d\n",i+1,nCombinations);
	  tuple_iterator(discrete, i);
	}
	bool changed = true;
	double time = -1;
	// choosing the optimal continuous values
	while(changed == true){
	  changed=false;
	  find_min(changed, time);
	}
	if(best_time < 0 || time < best_time) {
	  best_time = time;
	  best_index = i;
	}
      }

      // setting the best discrete index
      if(!discrete.empty())
	tuple_iterator(discrete, best_index);
      return best_time;
    } else {
      return call_back(call);
    }
  } 
}; 

template<class T, class...Targs>
InVar<T>* variable(T* var, std::string name, std::vector<T> values, bool continuous = true) {
  return new InVar<T>(var, name, values, continuous);
}
template<class T, class...Targs>
InVar<T>* variable(T* var, std::string name, T min =(T) 0, T max =(T) 0, T step =(T) 1, bool continuous = true) {
  std::vector<T> values;
  for(int i=0;i<=(int)((max-min)/step);i++) values.push_back((T)(min+step*i));
  return variable(var, name, values, continuous);
}
template<class...types>
SolverTimings<types...>* solverTimings(QUDA_solver& solver, PLEGMA_Vector<double>& vectorInOut, InVar<types>*... instance) {
  return new SolverTimings<types...>(solver,vectorInOut, instance...);
}
template<class T,class...types>
Minimizer<T,types...>* minimizer(T* call_back, bool enabled, InVar<types>*... instance) {
  return new Minimizer<T,types...>(call_back, enabled, instance...);
}
template<class T,class...types>
Minimizer<T,types...>* minimizer(T* call_back, InVar<types>*... instance) {
  return new Minimizer<T,types...>(call_back, true, instance...);
}

int main(int argc, char **argv) {
  std::vector<std::string> listOpt = {"load-gauge", "verbosity"};
  initializeOptions(argc, argv, true, listOpt);
  initializePLEGMA();

  {
    PLEGMA_Gauge<double> gauge;
    gauge.readFromLime(latfile.c_str());
    gauge.load();
    initGaugeQuda(gauge, true);
  }
    
  //Defining the QUDA_solver and executing the tuning with the "minimize" function
  {  
    PLEGMA_Vector<double> vectorIn;
    if(mu<0) mu*=-1.;
    QUDA_solver solver(mu);
    
    // set of parameters to tune with options
    auto mu_factor_ = variable(&mu_factor[mg_levels-1],"mu_factor",1.,101.,5.,true);
    auto coarse_solver_tol_ = variable(&coarse_solver_tol[mg_levels-1],"coarse solver tolerance",{0.01,0.022,0.046,0.1,0.22,0.46}, true);
    auto coarse_solver_ = variable(&coarse_solver[mg_levels-1],"Coarse solver", { QUDA_BICGSTAB_INVERTER, QUDA_GCR_INVERTER}, false);

    auto nu_pre_0 = variable(&nu_pre[0],"nu_pre_0",0,10,2, true);
    auto nu_post_0 = variable(&nu_post[0],"nu_post_0",1,10,1, true);
    auto schwarz_0 = variable(&schwarz_type[0],"schwarz_type_0",{QUDA_INVALID_SCHWARZ}, false);
    auto schwarz_cycle_0 = variable(&schwarz_cycle[0],"schwarz_cycle_0",1,4,1, true);
    auto smoother_tol_0 = variable(&smoother_tol[0],"smoother_tol_0",{0.01,0.022,0.046,0.1,0.22,0.46}, true);
    auto smoother_type_0 = variable(&smoother_type[0],"smoother_type_0", { QUDA_MR_INVERTER}, false);

    auto nu_pre_1 = variable(&nu_pre[1],"nu_pre_1",0,10,2, true);
    auto nu_post_1 = variable(&nu_post[1],"nu_post_1",1,10,1, true);
    auto schwarz_1 = variable(&schwarz_type[1],"schwarz_type_1",{QUDA_INVALID_SCHWARZ}, false);
    auto schwarz_cycle_1 = variable(&schwarz_cycle[1],"schwarz_cycle_1",1,4,1, true);
    auto smoother_tol_1 = variable(&smoother_tol[1],"smoother_tol_1", {0.01,0.022,0.046,0.1,0.22,0.46}, true);
    auto smoother_type_1 = variable(&smoother_type[1],"smoother_type_1", { QUDA_MR_INVERTER}, false);

    auto nvec_0 = variable(&nvec[0],"nvec_0",{24,32}, true);
    auto nvec_1 = variable(&nvec[1],"nvec_1",{24,32}, true);
    // block_0 and block_1 are not available yet
    auto block_0 = variable(&mg_block_volume[0],"block_0",{mg_block_volume[0]}, true);
    auto block_1 = variable(&mg_block_volume[1],"block_1",{mg_block_volume[1]}, true);

    // Solver which control the set of parameters
    auto solverT = solverTimings(solver, vectorIn, mu_factor_, coarse_solver_tol_, coarse_solver_,
				 nu_pre_0, nu_post_0, schwarz_0, schwarz_cycle_0, smoother_tol_0, smoother_type_0,
				 nu_pre_1, nu_post_1, schwarz_1, schwarz_cycle_1, smoother_tol_1, smoother_type_1,
				 nvec_0, nvec_1);

    // Splitting the parameters in smaller set and running nested minimizers
    // coarse, smoother_0, smoother_1 are indipendent minimizers calling solverT
    auto coarse = minimizer(solverT, mu_factor_, coarse_solver_tol_, coarse_solver_);
    
    auto smoother_0 = minimizer(solverT, nu_pre_0, nu_post_0, schwarz_0, schwarz_cycle_0, smoother_tol_0, smoother_type_0);
    auto smoother_1 = minimizer(solverT, (1 < mg_levels-1)? true : false, //enabled only if needed
				nu_pre_1, nu_post_1, schwarz_1, schwarz_cycle_1, smoother_tol_1, smoother_type_1);

    auto inner_params = std::make_tuple(coarse,smoother_0,smoother_1);

    // setup_1 minimizes coarse, smoother_0, smoother_1
    auto setup_1 = minimizer(&inner_params, (1 < mg_levels-1)? true : false, //enabled only if needed
			     block_1, nvec_1);

    // setup_0 minimizes setup_1
    auto setup_0 = minimizer(setup_1, block_0, nvec_0);

    setup_0->apply();
    solverT->printBest();
  }

  finalize();
  return 0;
}
