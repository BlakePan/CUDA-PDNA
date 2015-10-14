#include "Config.h"

__global__	void	setup_kernel			(curandState *state,int seed);
__global__	void	Init_Popu_kernel		(int* popu,curandState* state);
__global__	void	Init_Enzyme_kernel		(double* enzyme);
__global__	void	Evaluation_kernel		(int* popu,double* score,double* fitness,bool optima_selection=OPT);
__global__	void	Popu_sort_kernel		(int* popu,double* score,double* fitness);
__global__	void	Record_kernel			(double* score,double* record,int iter_counter);
__global__	void	Extract_kernel			(int* popu,int* schema_sample);
__global__	void	First_Filter_kernel		(int* popu,int* enzyme_sample);
__global__	void	Filter_kernel			(int* popu,int* enzyme_sample,bool optima_selection=OPT);
__global__	void	Update_Enzyme_kernel	(int* smp_enzyme,double* enzyme,int* enzyme_vec);
__global__	void	Update_glo_schema_kernel	(int* Schema_sample,double* global_schema,int* Enzyme_vec);
__global__	void	Update_loc_schema_kernel	(int* popu,double* Local_schema,double* fitness,int* Enzyme_vec);
__global__	void	Update_population			(int* popu,double* Global_schema,double* Local_schema,double* Search_schema,curandState *state);
__global__	void	Get_Entropy				(double* Search_schema,double* Entropy);
__global__	void	Migration				(double* Entropy,int* popu,curandState *state);
__host__ __device__	int*	Encode			(int amino,int* rand_num_ary);
__host__ __device__	double	TestProblem		(int* input_individual);
__host__ __device__	void	Decoder_real	(int* input_individual,double* upper,double* lower,double* output_vec);
__host__ __device__	void	Bobble_Sort		(double* target,int* index,int size);
__host__ __device__	void	Bobble_Sort_inv	(double* target,int* index,int size);
__host__ __device__	double	Evaluation		(int* input_individual,bool optima_selection);
__host__ __device__	void	Schema_Normalize(double* Schema);
__host__ __device__	double	Find_max		(double* data_ary,int sizeof_data,int& index);
__host__ __device__	double	Find_max		(double* data_ary,int sizeof_data);
__host__ __device__	double	Find_min		(double* data_ary,int sizeof_data);
__host__ __device__	int		Schema_base		(double* interval,int col,int randnum,double rand_bit);
__host__ __device__	void	Interval		(double* schema,double* interval);
__host__ __device__	double	GetEntropy		(double* data,int data_size);