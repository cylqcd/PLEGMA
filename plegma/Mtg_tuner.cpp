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
  std::vector<double> timing;

  InVar( T* variable, std::string name, T min =(T) 0, T max =(T) 0, T step =(T) 1): variable(variable), name(name){ 
    for(int i=0;i<=(int)((max-min)/step);i++) values.push_back((T)(min+step*i));
    }

  InVar( T* variable, std::string name, std::vector<T> values): variable(variable), name(name), values(values){ }
};

//Useful for unpacking tuple
//Taken from https://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer/7858971#7858971
template<int ...> struct seq {};

template<int N, int ...S> struct gens : gens<N-1, N-1, S...> {};

template<int ...S> struct gens<0, S...>{ typedef seq<S...> type; };

template<class ...types>
class InvTuning{

private:
  
  std::tuple<InVar<types>*...> Variables;
  std::vector<std::tuple<double,types...>> Timings;
  
public:

  InvTuning(InVar<types>*... var) {
    this->Variables = std::make_tuple(var...);
  }


  template<int...S>
  void set_data(double time,seq<S...>){
    Timings.push_back(std::make_tuple(time,*(std::get<S>(Variables)->variable)...));
  }

  void call_set_data(double time){
    set_data(time,typename gens<sizeof...(types)>::type());
  }
  
  template <typename T>
  void PLEGMA_MtgTest(QUDA_solver * solver, InVar<T>* param , PLEGMA_Vector<double>* vectorIn)
  {
    double t1, t0;
    PLEGMA_Vector<double> vectorOut;
    //solver->solve(vectorOut,*vectorIn);
    for(int i=0;i<param->values.size();i++)
      {
	*param->variable = param->values[i];
	solver->UpdateSolver();
	t1=MPI_Wtime();
	solver->solve(vectorOut,*vectorIn);
	t0 = MPI_Wtime()-t1;
	MPI_Allreduce(&t0, &t1, 1, MPI_Type(t0), MPI_MAX, MPI_COMM_WORLD);
	param->timing.push_back(t1);
	call_set_data(t1);
	PLEGMA_printf("NewParamSet: ");
	print(Timings[Timings.size()-1]);
	PLEGMA_printf("\n");
      }


    std::vector<double>::iterator result = std::min_element(std::begin(param->timing), std::end(param->timing));
    *param->variable = param->values[std::distance(std::begin(param->timing),result)];

    PLEGMA_printf("Setting the best value for the parameter: %s = %s\n",
		  param->name.c_str(),std::to_string(param->values[std::distance(std::begin(param->timing),result)]).c_str());

    solver->UpdateSolver(); 
  }
  

  //Method for iterating over tuple elements taken from:
  //https://stackoverflow.com/questions/1198260/how-can-you-iterate-over-the-elements-of-an-stdtuple
  
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  for_each(std::tuple<Tp...> &,QUDA_solver*, PLEGMA_Vector<double>*) 
  { }
  
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type
  for_each(std::tuple<Tp...>& t, QUDA_solver* solver, PLEGMA_Vector<double>*vec)
  {
    this->PLEGMA_MtgTest(solver,std::get<I>(t),vec);
    for_each<I + 1, Tp...>(t,solver,vec);
  }
    
    void MtgTune(QUDA_solver *solver, PLEGMA_Vector<double> *vec ){ for_each(this->Variables,solver,vec);}

  //Method for printing the tuple elements
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  print(std::tuple<Tp...>& t)
  { }

  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type
  print(std::tuple<Tp...>& t)
  {
    PLEGMA_printf("%s, ", std::to_string(std::get<I>(t)).c_str());
    print<I + 1, Tp...>(t);
  }
    
}; 

template<class...types>
InvTuning<types...> make_InvTuner(InVar<types>*... instance) {
  return InvTuning<types...>(instance...);
}


static std::vector<std::string> listOpt = {"load-gauge"};

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  initializePLEGMA();

  PLEGMA_Gauge<double> gauge;
  gauge.readFromLime(latfile.c_str());
  gauge.load();
  initGaugeQuda(gauge, true);

  // Setting the input vector field once
  PLEGMA_Vector<double> vectorIn;
  int (*sources);
  hostMalloc(sources, N_DIMS*sizeof(int));
  sources[0]=0;
  sources[1]=1;
  sources[2]=0;
  sources[3]=0;
  vectorIn.pointSource(sources, 0, 0, DEVICE);
  hostFree(sources);

  // Setting the Multigrid parameters one wants to test
  InVar<double> mu_factor_t(&mu_factor[mg_levels-1],"mu_factor_level",10.,70.,10.);

  InVar<int> nu_pre_t(&nu_pre,"nu_pre",0,6,2);
 
 
  InVar<int> nu_post_t(&nu_post,"nu_post",0,6,2);

  InVar<double> smoother_tol_0(&smoother_tol[0],"smoother tolerance 0",{0.01,0.04,0.1,0.4});

  InVar<double> smoother_tol_1(&smoother_tol[1],"smoother tolerance 1",{0.01,0.04,0.1,0.4});

  InVar<double> coarse_solver_tol_t(&coarse_solver_tol[mg_levels-1],"coarse solver tolerance",{0.01,0.04,0.1,0.4});


  InVar<QudaInverterType> smoother_type_0(&smoother_type[0],"Smoother solver level 0",{ QUDA_CGNE_INVERTER,
											QUDA_BICGSTAB_INVERTER,
											QUDA_GCR_INVERTER,
											QUDA_MR_INVERTER});

  InVar<QudaInverterType> smoother_type_1(&smoother_type[1],"Smoother solver level 1",{ QUDA_CGNE_INVERTER,
											QUDA_BICGSTAB_INVERTER,
											QUDA_GCR_INVERTER,
											QUDA_MR_INVERTER});
      
  InVar<QudaInverterType> coarse_type(&coarse_solver[mg_levels-1],"Coarse solver",{ QUDA_CGNE_INVERTER,
										    QUDA_BICGSTAB_INVERTER,
										    QUDA_GCR_INVERTER});


  //Defining the InvTuning class instance 
  auto tuner=make_InvTuner(&mu_factor_t,&nu_pre_t,&nu_post_t,
			   &smoother_tol_0,&smoother_tol_1,
			   &coarse_solver_tol_t,
			   &smoother_type_0,&smoother_type_1,&coarse_type); 
					

  //Defining the QUDA_solver and executing the tuning with the "MtgTune" function
  {  
    if(mu<0) mu*=-1.;
    QUDA_solver solver(mu);
    tuner.MtgTune(&solver,&vectorIn);
  }

  finalize();
  return 0;
}



