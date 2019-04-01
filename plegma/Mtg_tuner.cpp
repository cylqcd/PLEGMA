#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <list>

using namespace plegma;
using namespace quda;


int PLEGMA_printInvParams(QUDA_solver* solver);

template <typename T>
struct InVar{

  T *variable;
  std::vector<T> values;
  std::string name;
  bool continuous;
  
  InVar( T* variable, std::string name, T min =(T) 0, T max =(T) 0, T step =(T) 1 , bool continuous = true): variable(variable), name(name),continuous(continuous){ 
    for(int i=0;i<=(int)((max-min)/step);i++) values.push_back((T)(min+step*i));
  }

  InVar( T* variable, std::string name, std::vector<T> values, bool continuous = false): variable(variable), name(name), values(values), continuous(continuous){ }
};

//Useful for unpacking tuple
//Taken from https://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer/7858971#7858971
template<int ...> struct seq {};

template<int N, int ...S> struct gens : gens<N-1, N-1, S...> {};

template<int ...S> struct gens<0, S...>{ typedef seq<S...> type; };

template<class ...types>
class InvTuning{

protected:

  QUDA_solver &solver;
  PLEGMA_Vector<double> &vectorInOut;
  std::tuple<InVar<types>*...> Variables;
  std::vector<std::tuple<double,types...>> Timings;

  template<int...S>
  void set_data(double time,seq<S...>){
    Timings.push_back(std::make_tuple(time,*(std::get<S>(Variables)->variable)...));
  }
  void call_set_data(double time){
    set_data(time,typename gens<sizeof...(types)>::type());
  }

  template<std::size_t I = 0, typename time_t>
  inline typename std::enable_if<I == sizeof...(types), void>::type
  find_time(time_t &) {
  }
  template<std::size_t I = 0, typename time_t>
  inline typename std::enable_if<I < sizeof...(types), void>::type
  find_time(time_t &time) {
    auto value = *(std::get<I>(Variables)->variable);
    for (auto it = time.begin(); it < time.end(); ) {
      auto it_value = std::get<I+1>(*it);
      if(it_value != value)
	it = time.erase(it);
      else
	it++;
    }
    find_time<I+1>(time);
  }
    
  double solve(){
    auto time_copy = Timings;
    find_time(time_copy);
    if(time_copy.empty()) {
      double t0, t1;
      int sources[4]={0};
      sources[1]=1;
      vectorInOut.pointSource(sources, 0, 0, DEVICE);
      solver.UpdateSolver();
      t1=MPI_Wtime();
      solver.solve(vectorInOut, vectorInOut);
      t0 = MPI_Wtime()-t1;
      MPI_Allreduce(&t0, &t1, 1, MPI_Type(t0), MPI_MAX, MPI_COMM_WORLD);
      // Rescaling the time with the residual
      call_set_data(t1*log(tol)/log(solver.getSolverParam()->true_res));
      PLEGMA_printf("NewParamSet: ");
      print(Timings[Timings.size()-1]);
      PLEGMA_printf("\n");
      return t1;
    } else {
      return std::get<0>(time_copy[0]);
    }
  }
  
    
  template<std::size_t I = 0, typename disc_t>
  inline typename std::enable_if<I == sizeof...(types), void>::type
  select_discrete(disc_t& ) {
  }
  template<std::size_t I = 0, typename disc_t>
  inline typename std::enable_if<I < sizeof...(types), void>::type
  select_discrete(disc_t& discrete) {
    if (std::get<I>(Variables)->continuous == false) discrete.push_back({I, std::get<I>(Variables)->values.size()});
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
      auto elem=std::get<I>(Variables);
      *(elem->variable)= elem->values[i%discr[0][1]];
      i/=discr[0][1];
      discr.erase(discr.begin());
    }
    tuple_iterator<I+1>(discr,i);
  }


  template<std::size_t I = 0>
  inline typename std::enable_if<I == sizeof...(types), void>::type
  set_from_timings( size_t) {
  }
  template<std::size_t I = 0>
  inline typename std::enable_if<I < sizeof...(types), void>::type
  set_from_timings(size_t i) {
      auto elem=std::get<I>(Variables);
      *(elem->variable)= std::get<I+1>(Timings[i]);
      set_from_timings<I+1>(i);
  }

  
  template<std::size_t I = 0>
  inline typename std::enable_if<I == sizeof...(types), void>::type
  find_min(bool&) {
  }
  template<std::size_t I = 0>
  inline typename std::enable_if<I < sizeof...(types), void>::type
  find_min(bool &changed) {
    auto elem = std::get<I>(Variables);
    if(elem->continuous == true) {
      auto it = std::find(elem->values.begin(), elem->values.end(), *(elem->variable));
      if (it >= elem->values.end()) {
	it = elem->values.begin();
	*(elem->variable) = *it;
      }
      
      double t0 = solve();
      
      bool do_it=true;
      bool Ichanged=false;
      while (do_it && it+1 < elem->values.end()) {
	it++;
	*(elem->variable) = *it;
	double tp = solve();
	if(tp<t0) {
	  Ichanged=true;
	  changed=true;
	  t0=tp;
	} else {
	  do_it=false;
	  it--;
	  *(elem->variable) = *it;
	}
      }
      if(!Ichanged) {
	do_it=true;
	while (do_it && it-1 >= elem->values.begin()) {
	  it--;
	  *(elem->variable) = *it;
	  double tm = solve();
	  if(tm<t0) {
	    Ichanged=true;
	    changed=true;
	    t0=tm;
	  } else {
	    do_it=false;
	    it++;
	    *(elem->variable) = *it;
	  }
	}
      }
    }

    find_min<I+1>(changed);
  }

 
public:

  InvTuning(QUDA_solver &solver, PLEGMA_Vector<double> &vectorInOut, InVar<types>*... var) : solver(solver), vectorInOut(vectorInOut) {
    this->Variables = std::make_tuple(var...);
  }

  void minimize(){

    std::vector<std::array<size_t,2>> discrete;
    select_discrete(discrete);
    
    size_t nCombinations = 1;
    for(auto a : discrete) nCombinations*=a[1];
    
    int count = 0;
    PLEGMA_printf("NewParam: Set best discrete parameter\n");
    for(size_t i=0; i < nCombinations; i++){
      tuple_iterator(discrete, i);
      bool changed = true;
      while(changed == true){
	PLEGMA_printf("NewParam: iteration %d\n",count);
	count++;
	changed=false;
	find_min(changed);
      }
    }
    
    int best_index = 0;
    double best_time = std::get<0>(Timings[0]);
    for(size_t i=1; i<Timings.size(); i++) {
      if(best_time > std::get<0>(Timings[i])){ best_index = i; best_time = std::get<0>(Timings[i]);}
    }
    if(best_index != Timings.size()-1){
      set_from_timings(best_index);
      solver.UpdateSolver();
    }
    PLEGMA_printf("Set best set of discrete parameters: ");
    print(Timings[best_index]);
    PLEGMA_printf("\n");
  }
  
  //Method for printing the tuple elements
  template<std::size_t I = 0, typename ...Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  print(std::tuple<Tp...>& t) {
  }
  template<std::size_t I = 0, typename ...Tp>
  inline typename std::enable_if< (I > 0 & I < sizeof...(Tp)), void>::type
  print(std::tuple<Tp...>& t) {
    std::string name = (std::get<I-1>(Variables)->name);
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
    
}; 

template<class...types>
InvTuning<types...> make_InvTuner(QUDA_solver& solver, PLEGMA_Vector<double>& vectorInOut, InVar<types>*... instance) {
  return InvTuning<types...>(solver,vectorInOut,instance...);
}


static std::vector<std::string> listOpt = {"load-gauge"};

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  initializePLEGMA();

  {
    PLEGMA_Gauge<double> gauge;
    gauge.readFromLime(latfile.c_str());
    gauge.load();
    initGaugeQuda(gauge, true);
  }
    

  
  			   
  //Defining the QUDA_solver and executing the tuning with the "MtgTune" function
  {  
    PLEGMA_Vector<double> vectorIn;
    if(mu<0) mu*=-1.;
    QUDA_solver solver(mu);

    // tuning of coarsest level paramters
    {
      InVar<double> mu_factor_t(&mu_factor[mg_levels-1],"mu_factor_level",1.,70.,10.,true);
      InVar<double> coarse_solver_tol_t(&coarse_solver_tol[mg_levels-1],"coarse solver tolerance",{0.01,0.04,0.1,0.4}, true);
      InVar<QudaInverterType> coarse_type(&coarse_solver[mg_levels-1],"Coarse solver",{ QUDA_BICGSTAB_INVERTER,
											QUDA_GCR_INVERTER},false);
      auto coarse=make_InvTuner(solver,vectorIn,
				&mu_factor_t, &coarse_solver_tol_t, &coarse_type);
      coarse.minimize();
    }

    // tuning of smoother paramters
    for(int level=0; level < mg_levels-1; level++) {
      InVar<int> nu_pre_t(&nu_pre[level],"nu_pre_"+std::to_string(level),0,10,2, true);
      InVar<int> nu_post_t(&nu_post[level],"nu_post_"+std::to_string(level),0,10,2, true);
      InVar<QudaSchwarzType> schwarz_t(&schwarz_type[level],"schwarz_"+std::to_string(level),{QUDA_INVALID_SCHWARZ,
											      QUDA_ADDITIVE_SCHWARZ}, false);
      InVar<int> schwarz_cycle_t(&schwarz_cycle[level],"schwarz_cycle"+std::to_string(level),1,4,1, true);
      InVar<double> smoother_tol_t(&smoother_tol[level],"smoother_tol_"+std::to_string(level),{0.01,0.04,0.1,0.4}, true);
      InVar<QudaInverterType> smoother_type_t(&smoother_type[level],"smoother_type_"+std::to_string(level),{ QUDA_BICGSTAB_INVERTER,
													     QUDA_GCR_INVERTER,
													     QUDA_MR_INVERTER }, false);
      auto smoother=make_InvTuner(solver,vectorIn,
				  &nu_pre_t,&nu_post_t,
				  &schwarz_t, &schwarz_cycle_t,
				  &smoother_tol_t,&smoother_type_t);      
      smoother.minimize();
    }
  }

  finalize();
  return 0;
}



