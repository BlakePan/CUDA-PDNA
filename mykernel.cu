#ifndef	_MY_KERNEL_H_
#define _MY_KERNEL_H_

#include "mykernel.cuh"

__global__	void	setup_kernel			(curandState *state,int seed)
{
	/*
		Description:
		Set up cuda random state.

		Kernel Dimesion:
		<<<TUBE,POPULATION>>>

		Input:
		cuda rand state.
		seed.		
	*/

	int id = threadIdx.x + blockIdx.x * POPULATION;	
	curand_init(seed, id, 0, &state[id]);

	if(blockIdx.x==0 && threadIdx.x==0){
		for(int d=0;d<CODON_DIGIT;d++){
			_AMINO_RANGE+=20*pow(21.0,d);
		}
	}	
}
__global__	void	Init_Popu_kernel		(int* popu,curandState* state)
{
	/*
		Description:
		Initialize populations.

		Kernel Dimesion:
		<<<TUBE,POPULATION>>>

		Input:
		population pointer.
		curand state.		
	*/
	int tube = blockIdx.x;
	int	indi = threadIdx.x;
	int	id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState local_state = state[id];
	__shared__	int amino_range;
	amino_range = _AMINO_RANGE;

	for(int i=0;i<FIT_LENGTH;i+=CODON_LENGTH){
		int animo = curand(&local_state)%amino_range;
		int rand_num_ary[CODON_DIGIT] = {};
					
		for(int c=0;c<CODON_DIGIT;c++){						
			rand_num_ary[c] = curand(&local_state);
		}

		int* code = Encode(animo,rand_num_ary);

		for(int j=0;j<CODON_LENGTH;j++){			
			*(popu + indi * (TUBE*FIT_LENGTH) + tube*FIT_LENGTH+i+j)
				= *(code+j);
		}		
	}
	__syncthreads();

}
__global__	void	Init_Enzyme_kernel		(double* enz)
{
	/*
		Description:
		Initialize restrict enzyme.

		Kernel Dimesion:
		<<<1,1>>>

		Input:
		restrict enzyme pointer.				
	*/
	for(int i=0;i<4;i++){
		for(int j=0;j<FIT_LENGTH;j++){
			*(enz + i*FIT_LENGTH +j) = 0.25;
		}
	}
}
__global__	void	Evaluation_kernel		(int* popu,double* score,double* fit,bool opt)
{
	/*
		Description:
		Evaluate all individaul.
		Calculate their scores and fitnesses.

		Kernel Dimesion:
		<<<TUBE,POPULATION>>>

		Input:
		population pointer.
		score pointer.		
		fitness pointer.
		switch to find min or max.
	*/
	int	tube = blockIdx.x;
	int	indi = threadIdx.x;
	int	local_indi[FIT_LENGTH]={};	
	//double	crent_axis[DIMENSION]={};
	double	local_score = 0;
	double	local_fit = 0;

	//Copy individual from global to local
	for(int i=0;i<FIT_LENGTH;i++){
		local_indi[i] = *(popu + indi * (TUBE*FIT_LENGTH) + tube*FIT_LENGTH+i);
	}
	__syncthreads();

	//Calculate scores and fitnesses
	local_score = TestProblem(local_indi);
	if(opt){//find min
		local_fit = -local_score;
	}else{//find max
		local_fit = local_score;
	}
	__syncthreads();

	//Output 
	*(score + indi * TUBE + tube) = local_score;
	*(fit + indi * TUBE + tube) = local_fit;
	__syncthreads();
}
__global__	void	Popu_sort_kernel		(int* popu,double* score,double* fit)
{
	/*
		Description:
		Sort population accordding to its fitness.

		Kernel Dimesion:
		<<<1,TUBE>>>

		Input:
		unsorted population pointer.
		unsorted score pointer.		
		unsorted fitness pointer.
	*/
	//int		tube = blockIdx.x;
	int		tube = threadIdx.x;
	int		local_popu[POPULATION][FIT_LENGTH]	={};
	double	local_score[POPULATION]				={};
	double	local_fit[POPULATION]				={};

	//Copy from global to local
	for(int i=0;i<POPULATION;i++){
		for(int j=0;j<FIT_LENGTH;j++){
			local_popu[i][j] = 
				*(popu + i* TUBE*FIT_LENGTH + j+tube*FIT_LENGTH);
		}
	}
	__syncthreads();

	for(int i=0;i<POPULATION;i++){
		local_score[i] = *(score + i*TUBE + tube);
	}

	for(int i=0;i<POPULATION;i++){
		local_fit[i] = *(fit + i*TUBE + tube);
	}
	__syncthreads();

	//Sorting
	int index[POPULATION] = {};
	for(int i=0;i<POPULATION;i++){
		index[i] = i;
	}
	Bobble_Sort_inv(local_fit,index,POPULATION);	
	__syncthreads();

	//Output
	for(int i=0;i<POPULATION;i++){
		*(fit + i*TUBE + tube) = local_fit[i];
	}

	for(int i=0;i<POPULATION;i++){
		*(score + i*TUBE + tube) = local_score[index[i]];
	}

	for(int i=0;i<POPULATION;i++){
		for(int j=0;j<FIT_LENGTH;j++){
			*(popu + i* TUBE*FIT_LENGTH + j+tube*FIT_LENGTH) 
				= local_popu[index[i]][j];
		}
	}
	__syncthreads();
	
}
__global__	void	Record_kernel			(double* score,double* rec,int iter)
{
	/*
		Description:
		Record best fitness in each iter in each tube.

		Kernel Dimesion:
		<<<1,TUBE>>>

		Input:		
		unsorted score pointer.		
		recorder pointer.
		current iteration.
	*/
	int tube = threadIdx.x;
	
	*(rec + iter * TUBE + tube) = *(score + tube);
	__syncthreads();
}
__global__	void	Extract_kernel			(int* popu,int* sample)
{
	/*
		Description:
		Extract top individudals in each tube.
		Make them be sample data of schemata.

		Kernel Dimesion:
		<<<1,TUBE>>>

		Input:		
		sorted population pointer.		
		sample data pointer.
	*/
	int tube = threadIdx.x;
	for(int i=0;i<FIT_LENGTH;i++){
		*(sample + tube * FIT_LENGTH + i) = 
			*(popu + tube * FIT_LENGTH + i);
	}
	

	/*debug
	if(threadIdx.x==0){
		printf("sample\n");
		for(int t=0;t<TUBE;t++){
			for(int i=0;i<FIT_LENGTH;i++){
				printf("%d ",*(sample + t* FIT_LENGTH +i));
			}			
			printf("\n");
		}
		printf("\n");

		printf("population\n");
		for(int t=0;t<TUBE;t++){
			for(int p=0;p<POPULATION;p++){
				for(int i=0;i<FIT_LENGTH;i++){
					printf("%d ",*(popu + p*TUBE*FIT_LENGTH + i+t*FIT_LENGTH));
				}
				printf("\n");
			}					
			printf("\n");
		}
		printf("\n");
	}*/
	__syncthreads();
	
}
__global__	void	First_Filter_kernel		(int* popu,int* sample)
{
	/*
		Description:
		Extract worst individudals in each tube.
		Make them to be sample datas of enzyme.		

		Kernel Dimesion:
		<<<1,TUBE>>>

		Input:		
		sorted population pointer.		
		sample data pointer.		
	*/
	int tube = threadIdx.x;
	for(int i=0;i<FIT_LENGTH;i++){
		*(sample + tube * FIT_LENGTH + i) = 
			*(popu + (POPULATION-1) * TUBE * FIT_LENGTH + tube * FIT_LENGTH + i);
	}

	__syncthreads();

}
__global__	void	Filter_kernel			(int* popu,int* sample,bool opt)
{
	/*
		Description:
		Extract worst individudals in each tube.
		Compare to sample data in last iteration,
		if find worse individual,
		update sample data.

		Kernel Dimesion:
		<<<1,1>>>

		Input:		
		sorted population pointer.		
		sample data pointer.	
		switch to find min or max.
	*/
	int	sample_data[TUBE*2][FIT_LENGTH];
	for(int i=0;i<1*TUBE*2;i++){
		for(int j=0;j<FIT_LENGTH;j++){
			sample_data[i][j] = -1;
		}
	}

	//Get sample data in last iteration
	int	data_cnt = 0;
	for(int i=0;i<TUBE;i++){
		for(int s=0;s<1;s++){
			for(int j=0;j<FIT_LENGTH;j++){
				sample_data[data_cnt][j] = 
					*(sample + s*FIT_LENGTH*TUBE +j+i*FIT_LENGTH);
			}
			data_cnt++;
		}		
	}

	//Get new sample data from current iteration
	for(int i=0;i<TUBE;i++){
		for(int s=0;s<1;s++){
			for(int j=0;j<FIT_LENGTH;j++){
				sample_data[data_cnt][j] = 
					*(popu + (POPULATION-s-1)*FIT_LENGTH*TUBE + j+i*FIT_LENGTH );
			}
			data_cnt++;
		}
	}

	//Evaluate and Sort the data 
	int		index[1*TUBE*2];
	double	fit[1*TUBE*2];

	for(int i=0;i<1*TUBE*2;i++){
		index[i] = i;
	}

	for(int i=0;i<1*TUBE*2;i++){
		int indi_tmp[FIT_LENGTH]={};
		for(int j=0;j<FIT_LENGTH;j++){
			indi_tmp[j] = sample_data[i][j];
		}
		fit[i] = Evaluation(indi_tmp,opt);
	}

	Bobble_Sort(fit,index,1*TUBE*2);	

	//Output:Keep the first half of the sample data
	for(int i=0;i<1*TUBE;i++){
		int keep_index = index[i];
		for(int j=0;j<FIT_LENGTH;j++){
			*(sample + i*FIT_LENGTH + j) = 
				sample_data[keep_index][j];
		}
	}
	
}
__global__	void	Update_Enzyme_kernel	(int* smp_enzyme,double* enzyme,int* enzyme_vec)
{
	/*
		Description:
		Update restrict enzyme by sample data.

		Kernel Dimesion:
		<<<1,1>>>

		Input:				
		sample data.		
		restrict enzyme.
		enzyme vector.
	*/

	double	share_enzyme[4][FIT_LENGTH]={};
	int		share_sample[1*TUBE][FIT_LENGTH]={};
	int		tmp_enzyme_vec[FIT_LENGTH];
	
	//Copy sample data from global to shared memory
	int data_cnt = 0;

	for(int i=0;i<TUBE;i++){
		for(int s=0;s<1;s++){
			for(int j=0;j<FIT_LENGTH;j++){
				share_sample[data_cnt][j] = 
					*(smp_enzyme + s*TUBE*FIT_LENGTH +j+i*FIT_LENGTH);
			}
			data_cnt++;
		}
	}	

	//Update enzyme
	for(int i=0;i<1*TUBE;i++){
		for(int j=0;j<FIT_LENGTH;j++){
			int base = share_sample[i][j];
			share_enzyme[base][j]++;
		}
	}

	//Normalization
	Schema_Normalize(&share_enzyme[0][0]);	
		
	//Make enzyme in vector form
	for(int j=0;j<FIT_LENGTH;j++){
		double tmp_data[4];
		int		cutter=-1;

		for(int i=0;i<4;i++){
			tmp_data[i] = share_enzyme[i][j];
		}
		if(Find_max(tmp_data,4,cutter) >= ENZ_MASK_TH){
			tmp_enzyme_vec[j] = cutter;
		}else{
			tmp_enzyme_vec[j] = -1;
		}
	}	

	//Output
	for(int i=0;i<4;i++){
		for(int j=0;j<FIT_LENGTH;j++){
			*(enzyme + i*FIT_LENGTH + j) = share_enzyme[i][j];
		}
	}

	for(int j=0;j<FIT_LENGTH;j++){
		*(enzyme_vec +j) = tmp_enzyme_vec[j];
	}

}
__global__	void	Update_glo_schema_kernel	(int* Schema_sample,double* global_schema,int* Enzyme_vec)
{
	/*
		Description:
		Update global schemas with sample datas

		Kernel Dimesion:
		<<<1,1>>>

		Input:				
		sample data sets.
		schema memories.
		enzyme vector.
	*/

	int		sample[TUBE][FIT_LENGTH];	
	int		enz_vec[FIT_LENGTH]={};	
	double	temp_schema[4][FIT_LENGTH]={};	

	//Copy memories
	for(int t=0;t<TUBE;t++){
		for(int i=0;i<FIT_LENGTH;i++){
			sample[t][i] = *(Schema_sample + t*FIT_LENGTH +i);
		}
	}
	
	for(int i=0;i<FIT_LENGTH;i++){
		enz_vec[i] = *(Enzyme_vec + i);
	}
	
	//Make temp schema
	for(int t=0;t<TUBE;t++){
		for(int i=0;i<FIT_LENGTH;i++){
			int base = sample[t][i];
			int	cutter = enz_vec[i];
			if(cutter<0 || cutter!=base){//dont cut
				temp_schema[base][i]++;
			}
		}
		
	}
	Schema_Normalize(&temp_schema[0][0]);

	//Output
	for(int i=0;i<4;i++){
		for(int j=0;j<FIT_LENGTH;j++){
			global_schema[(i*FIT_LENGTH) + j] 
			= temp_schema[i][j];
		}
	}

}
__global__	void	Update_loc_schema_kernel	(int* popu,double* Local_schema,double* Fit,int* Enzyme_vec)
{
	/*
		Description:
		Update local schemas with sample data

		Kernel Dimesion:
		<<<1,TUBE>>>

		Input:				
		population.
		schema memories.
		enzyme vector.
	*/

	int	tube = threadIdx.x;
	int	local_popu[POPULATION][FIT_LENGTH]={};
	int	enz_vec[FIT_LENGTH]={};
	double	fitness[POPULATION]={};
	double	temp_schema[4][FIT_LENGTH]={};
	double sum,ave;

	//Copy population to local
	for(int i=0;i<POPULATION;i++){
		for(int j=0;j<FIT_LENGTH;j++){
			local_popu[i][j] = *(popu + i*FIT_LENGTH +j);
		}
	}

	//Copy enzyme vector
	for(int i=0;i<FIT_LENGTH;i++){
		enz_vec[i] = *(Enzyme_vec + i);
	}

	//Copy Fitness
	for(int i=0;i<POPULATION;i++){
		fitness[i] = *(Fit + i*TUBE +tube);
	}
	__syncthreads();

	sum=0;
	for(int i=0;i<POPULATION;i++){
		sum+=fitness[i];
	}
	ave = sum/POPULATION;
	__syncthreads();

	//Make temp schema
	for(int p=0;p<POPULATION;p++){
		for(int i=0;i<FIT_LENGTH;i++){
			int base = local_popu[p][i];
			int	cutter = enz_vec[i];
			if(cutter<0 || cutter!=base){//dont cut
				temp_schema[base][i]+=ROUND(fitness[p]/ave);
			}
		}
	}	
	__syncthreads();

	Schema_Normalize(&temp_schema[0][0]);
	__syncthreads();

	//Output
	for(int i=0;i<4;i++){
		for(int j=0;j<FIT_LENGTH;j++){
			Local_schema[(i*FIT_LENGTH*TUBE) + j+tube*FIT_LENGTH] 
			= temp_schema[i][j];
		}
	}
	__syncthreads();
}
__global__	void	Update_population			(int* popu,double* Global_schema,double* Local_schema,
												 double* Search_schema,curandState *state)
{
	/*
		Description:
		Combine Global and Local Schema,
		And Update population by Search Schema.

		Kernel Dimesion:
		<<<TUBE,POPULATION>>>

		Input:				
		population.
		Global Schema.
		Local Schema.
		Search Schema.
		curand state.
	*/
	int tube = blockIdx.x;
	int	indi = threadIdx.x;
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int	local_indi[FIT_LENGTH]={};
	double	local_schema[4][FIT_LENGTH]={};	
	double	global_schema[4][FIT_LENGTH]={};

	__shared__	double	search_schema[4][FIT_LENGTH];
	__shared__	double	search_schema_intvl[4][FIT_LENGTH];
	__shared__	int		amino_range;

	curandState local_state = state[id];

	//Copy schemata
	for(int i=0;i<4;i++){
		for(int j=0;j<FIT_LENGTH;j++){
			global_schema[i][j] =*(Global_schema + i*FIT_LENGTH + j);
		}
	}
	for(int i=0;i<4;i++){
		for(int j=0;j<FIT_LENGTH;j++){
			local_schema[i][j] = *(Local_schema + i*FIT_LENGTH*TUBE +j+tube*FIT_LENGTH);
		}
	}
	__syncthreads();

	//Make Search schema
	if(threadIdx.x==0){
		for(int i=0;i<4;i++){
			for(int j=0;j<FIT_LENGTH;j++){
				search_schema[i][j] = 
					COMB_GLO*global_schema[i][j] + COMB_LOC*local_schema[i][j];
			}
		}

		Interval(&search_schema[0][0],&search_schema_intvl[0][0]);

		amino_range = _AMINO_RANGE;
	}
	__syncthreads();

	//Update population
	if(threadIdx.x==0){//Elitism
		for(int i=0;i<FIT_LENGTH;i++){
			local_indi[i] = *(popu + i+tube*FIT_LENGTH);
		}
	}else if(threadIdx.x<=POPU_SCHEMA){//Update by schema
		for(int i=0;i<FIT_LENGTH;i++){
			double	rand_bit = curand_uniform_double(&local_state);
			int		rand_num = curand(&local_state)%4;
			int		base = Schema_base(&search_schema_intvl[0][0],i,rand_num,rand_bit);
			local_indi[i]= base;
		}
	}else{//randomly generate
		for(int i=0;i<FIT_LENGTH;i+=CODON_LENGTH){
			int animo = curand(&local_state)%amino_range;
			int rand_num_ary[CODON_DIGIT] = {};
						
			for(int c=0;c<CODON_DIGIT;c++){						
				rand_num_ary[c] = curand(&local_state);
			}

			int* code = Encode(animo,rand_num_ary);

			for(int j=0;j<CODON_LENGTH;j++){			
				local_indi[i+j] = *(code+j);
			}
		}
		
	}		
	__syncthreads();

	//Output
	state[id] = local_state;

	for(int i=0;i<FIT_LENGTH;i++){
		*(popu + indi*FIT_LENGTH*TUBE + i+tube*FIT_LENGTH)=
			local_indi[i];
	}
	__syncthreads();
	
}
__global__	void	Get_Entropy					(double* Search_schema,double* Entropy)
{
	/*
		Description:
		Calculate Entropy of a Search Scheam.

		Kernel Dimesion:
		<<<TUBE,FIT_LENGTH>>>

		Input:				
		Search schema.
		entropy vector.
	*/
	
	int tube = blockIdx.x;
	int	bit = threadIdx.x;
	double	data_set[4]={};
	double	out_entropy = 0;
	__shared__	double tmp_entropy[FIT_LENGTH];

	tmp_entropy[bit]=0;
	__syncthreads();

	for(int i=0;i<4;i++){
		data_set[i] = *(Search_schema + i*FIT_LENGTH*TUBE + bit+tube*FIT_LENGTH);
	}
	__syncthreads();

	tmp_entropy[bit] = GetEntropy(data_set,4);
	__syncthreads();

	if(threadIdx.x==0){
		for(int i=0;i<FIT_LENGTH;i++){
			out_entropy+=tmp_entropy[i];
		}
	}
	__syncthreads();

	//Output
	*(Entropy + tube) = out_entropy;
	
}
__global__	void	Migration					(double* Entropy,int* popu,curandState *state)
{
	/*
		Description:
		If a tube of which entropy is lower than the th-migrate,
		activate Migration.

		Kernel Dimesion:
		<<<TUBE,POPULATION>>>

		Input:				
		entropy vector.
		population.
		curand state.
	*/

	int	Trasl_table[4][4][4]={ {{1,1,2,2},{3,3,3,3},{4,4,0,0},{5,5,0,6}} ,
							   {{2,2,2,2},{7,7,7,7},{8,8,9,9},{10,10,10,10}} ,
							   {{11,11,12,12},{13,13,13,13},{14,14,15,15},{3,3,10,10}} ,
							   {{16,16,16,16},{17,17,17,17},{18,18,19,19},{20,20,20,20}}};
	int tube = blockIdx.x;
	int	indi = threadIdx.x;
	int	id = threadIdx.x + blockIdx.x*blockDim.x;
	int	local_indi[FIT_LENGTH]={};
	__shared__	int		best_indi[FIT_LENGTH];
	__shared__	int		amino_range;
	curandState local_state = state[id];

	if(*(Entropy+tube) <= MIGRAT_TH){//Activate Migration!
		if(threadIdx.x==0){
			for(int i=0;i<FIT_LENGTH;i++){
				best_indi[i] = *(popu + i+tube*FIT_LENGTH);
			}
			amino_range = _AMINO_RANGE;
		}
		__syncthreads();

		if(threadIdx.x!=0){//if not the best
			//copy individual
			for(int i=0;i<FIT_LENGTH;i++){
				local_indi[i] = *(popu + indi*FIT_LENGTH*TUBE +i+tube*FIT_LENGTH);
			}
			
			for(int d=0;d<DIMENSION;d++){//check each dimemsion
				int	digit_code[CODON_LENGTH]={};
				int	crent_amino = 0;

				int	best_digit_code[CODON_LENGTH]={};										
				int	best_amino = 0;			

				//Decode in dimension d
				for(int c=0;c<CODON_LENGTH;c++){
					digit_code[c]		=	local_indi[d*CODON_LENGTH+c];
					best_digit_code[c]	=	best_indi[d*CODON_LENGTH+c];
				}
				__syncthreads();

				//Decode to animo
				for(int i=0;i<CODON_DIGIT;i++){//for each digit
					int index = 0+i*3;
					crent_amino+=Trasl_table[digit_code[index]][digit_code[index+1]][digit_code[index+2]] *pow(21.0,i);
					best_amino+=Trasl_table[best_digit_code[index]][best_digit_code[index+1]][best_digit_code[index+2]] *pow(21.0,i);					
				}
				__syncthreads();

				//Calculate error
				double err = abs(best_amino-crent_amino);	

				if(err<=MIGRAT_ERR_TOL){//Extend in dimension d
					//Decide direction
					if(curand_uniform_double(&local_state) > (double)crent_amino/amino_range){//increase
						crent_amino = best_amino + curand_uniform_double(&local_state) * (amino_range - best_amino);						
					}else{//decrease
						crent_amino = best_amino + curand_uniform_double(&local_state) * (0 - best_amino);						
					}		
					__syncthreads();

					//Encode
					int rand_num_ary[CODON_DIGIT] = {};
					for(int i=0;i<CODON_DIGIT;i++){
						rand_num_ary[i] = curand(&local_state);
					}
					int* new_code = Encode(crent_amino,rand_num_ary);		
					__syncthreads();

					for(int c=0;c<CODON_LENGTH;c++){
						local_indi[d*CODON_LENGTH+c] = *(new_code+c);
					}
					__syncthreads();
				}
			}//end of each dimension
			__syncthreads();

			//copy individual back to populatopn
			for(int i=0;i<FIT_LENGTH;i++){
				*(popu + indi*FIT_LENGTH*TUBE +i+tube*FIT_LENGTH) = local_indi[i];
			}
			__syncthreads();
		}//end of if not the best
		__syncthreads();
	}//end of Migration
	__syncthreads();

	state[id] = local_state;
	__syncthreads();
}
__host__ __device__	int*	Encode					(int tar,int* rand_num_ary)
{
	int code[CODON_LENGTH]={};
	int digit[CODON_DIGIT]={};
	int sel=0;
	int tmp = tar;

	for(int i=0;i<CODON_DIGIT;i++){		
		digit[i] = tmp%21;
		tmp = tar/21;
	}

	for(int i=0;i<CODON_DIGIT;i++){
		int digit_tmp = digit[i];
		int code_tmp[3]={};
		int rand_num = rand_num_ary[i];
		switch(digit_tmp){
			case 0:
				sel = rand_num%3;
				if(sel==0){
					code_tmp[0] = 0;
					code_tmp[1] = 2;
					code_tmp[2] = 2;
				}else if(sel==1){
					code_tmp[0] = 0;
					code_tmp[1] = 2;
					code_tmp[2] = 3;
				}else{
					code_tmp[0] = 0;
					code_tmp[1] = 3;
					code_tmp[2] = 2;
				}
				break;

			case 1:
				sel = rand_num%2;
				if(sel==0){
					code_tmp[0] = 0;
					code_tmp[1] = 0;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 0;
					code_tmp[1] = 0;
					code_tmp[2] = 1;
				}
				break;

			case 2:
				sel = rand_num%6;
				if(sel==0){
					code_tmp[0] = 0;
					code_tmp[1] = 0;
					code_tmp[2] = 2;
				}else if(sel==1){
					code_tmp[0] = 0;
					code_tmp[1] = 0;
					code_tmp[2] = 3;
				}else if(sel==2){
					code_tmp[0] = 1;
					code_tmp[1] = 0;
					code_tmp[2] = 0;
				}else if(sel==3){
					code_tmp[0] = 1;
					code_tmp[1] = 0;
					code_tmp[2] = 1;
				}else if(sel==4){
					code_tmp[0] = 1;
					code_tmp[1] = 0;
					code_tmp[2] = 2;
				}else{
					code_tmp[0] = 1;
					code_tmp[1] = 0;
					code_tmp[2] = 3;
				}
				break;
				
			case 3:
				sel = rand_num%6;
				if(sel==0){
					code_tmp[0] = 0;
					code_tmp[1] = 1;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 0;
					code_tmp[1] = 1;
					code_tmp[2] = 1;
				}else if(sel==2){
					code_tmp[0] = 0;
					code_tmp[1] = 1;
					code_tmp[2] = 2;
				}else if(sel==3){
					code_tmp[0] = 0;
					code_tmp[1] = 1;
					code_tmp[2] = 3;
				}else if(sel==4){
					code_tmp[0] = 2;
					code_tmp[1] = 3;
					code_tmp[2] = 0;
				}else{
					code_tmp[0] = 2;
					code_tmp[1] = 3;
					code_tmp[2] = 1;
				}
				break;

			case 4:
				sel = rand_num%2;
				if(sel==0){
					code_tmp[0] = 0;
					code_tmp[1] = 2;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 0;
					code_tmp[1] = 2;
					code_tmp[2] = 1;
				}
				break;

			case 5:
				sel = rand_num%2;
				if(sel==0){
					code_tmp[0] = 0;
					code_tmp[1] = 3;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 0;
					code_tmp[1] = 3;
					code_tmp[2] = 1;
				}
				break;

			case 6:
				code_tmp[0] = 0;
				code_tmp[1] = 3;
				code_tmp[2] = 3;
				break;

			case 7:
				sel = rand_num%4;
				if(sel==0){
					code_tmp[0] = 1;
					code_tmp[1] = 1;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 1;
					code_tmp[1] = 1;
					code_tmp[2] = 1;
				}else if(sel==2){
					code_tmp[0] = 1;
					code_tmp[1] = 1;
					code_tmp[2] = 2;
				}else{
					code_tmp[0] = 1;
					code_tmp[1] = 1;
					code_tmp[2] = 3;
				}
				break;

			case 8:
				sel = rand_num%2;
				if(sel==0){
					code_tmp[0] = 1;
					code_tmp[1] = 2;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 1;
					code_tmp[1] = 2;
					code_tmp[2] = 1;
				}
				break;

			case 9:
				sel = rand_num%2;
				if(sel==0){
					code_tmp[0] = 1;
					code_tmp[1] = 2;
					code_tmp[2] = 2;
				}else if(sel==1){
					code_tmp[0] = 1;
					code_tmp[1] = 2;
					code_tmp[2] = 3;
				}
				break;

			case 10:
				sel = rand_num%6;
				if(sel==0){
					code_tmp[0] = 1;
					code_tmp[1] = 3;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 1;
					code_tmp[1] = 3;
					code_tmp[2] = 1;
				}else if(sel==2){
					code_tmp[0] = 1;
					code_tmp[1] = 3;
					code_tmp[2] = 2;
				}else if(sel==3){
					code_tmp[0] = 1;
					code_tmp[1] = 3;
					code_tmp[2] = 3;
				}else if(sel==4){
					code_tmp[0] = 2;
					code_tmp[1] = 3;
					code_tmp[2] = 2;
				}else{
					code_tmp[0] = 2;
					code_tmp[1] = 3;
					code_tmp[2] = 3;
				}
				break;

			case 11:
				sel = rand_num%2;
				if(sel==0){
					code_tmp[0] = 2;
					code_tmp[1] = 0;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 2;
					code_tmp[1] = 0;
					code_tmp[2] = 1;
				}
				break;

			case 12:
				sel = rand_num%2;
				if(sel==0){
					code_tmp[0] = 2;
					code_tmp[1] = 0;
					code_tmp[2] = 2;
				}else if(sel==1){
					code_tmp[0] = 2;
					code_tmp[1] = 0;
					code_tmp[2] = 3;
				}
				break;

			case 13:
				sel = rand_num%4;
				if(sel==0){
					code_tmp[0] = 2;
					code_tmp[1] = 1;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 2;
					code_tmp[1] = 1;
					code_tmp[2] = 1;
				}else if(sel==2){
					code_tmp[0] = 2;
					code_tmp[1] = 1;
					code_tmp[2] = 2;
				}else{
					code_tmp[0] = 2;
					code_tmp[1] = 1;
					code_tmp[2] = 3;
				}
				break;

			case 14:
				sel = rand_num%2;
				if(sel==0){
					code_tmp[0] = 2;
					code_tmp[1] = 2;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 2;
					code_tmp[1] = 2;
					code_tmp[2] = 1;
				}
				break;

			case 15:
				sel = rand_num%2;
				if(sel==0){
					code_tmp[0] = 2;
					code_tmp[1] = 2;
					code_tmp[2] = 2;
				}else if(sel==1){
					code_tmp[0] = 2;
					code_tmp[1] = 2;
					code_tmp[2] = 3;
				}
				break;

			case 16:
				sel = rand_num%4;
				if(sel==0){
					code_tmp[0] = 3;
					code_tmp[1] = 0;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 3;
					code_tmp[1] = 0;
					code_tmp[2] = 1;
				}else if(sel==2){
					code_tmp[0] = 3;
					code_tmp[1] = 0;
					code_tmp[2] = 2;
				}else{
					code_tmp[0] = 3;
					code_tmp[1] = 0;
					code_tmp[2] = 3;
				}
				break;

			case 17:
				sel = rand_num%4;
				if(sel==0){
					code_tmp[0] = 3;
					code_tmp[1] = 1;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 3;
					code_tmp[1] = 1;
					code_tmp[2] = 1;
				}else if(sel==2){
					code_tmp[0] = 3;
					code_tmp[1] = 1;
					code_tmp[2] = 2;
				}else{
					code_tmp[0] = 3;
					code_tmp[1] = 1;
					code_tmp[2] = 3;
				}
				break;

			case 18:
				sel = rand_num%2;
				if(sel==0){
					code_tmp[0] = 3;
					code_tmp[1] = 2;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 3;
					code_tmp[1] = 2;
					code_tmp[2] = 1;
				}
				break;

			case 19:
				sel = rand_num%2;
				if(sel==0){
					code_tmp[0] = 3;
					code_tmp[1] = 2;
					code_tmp[2] = 2;
				}else if(sel==1){
					code_tmp[0] = 3;
					code_tmp[1] = 2;
					code_tmp[2] = 3;
				}
				break;

			case 20:
				sel = rand_num%4;
				if(sel==0){
					code_tmp[0] = 3;
					code_tmp[1] = 3;
					code_tmp[2] = 0;
				}else if(sel==1){
					code_tmp[0] = 3;
					code_tmp[1] = 3;
					code_tmp[2] = 1;
				}else if(sel==2){
					code_tmp[0] = 3;
					code_tmp[1] = 3;
					code_tmp[2] = 2;
				}else{
					code_tmp[0] = 3;
					code_tmp[1] = 3;
					code_tmp[2] = 3;
				}
				break;

		}

		for(int j=0;j<3;j++){
			code[i*3+j] = code_tmp[j];
		}

	}

	return code;
}
__host__ __device__	double	TestProblem				(int* indi)
{
	/*
		Description:
		Test problems

		Input:
		dna individual.

		Output:
		score.
	*/	

	//upper bound and lower bound
	double	upper[DIMENSION] = {};
	double	lower[DIMENSION] = {};	
	
	double	result = 0;
	double	x[DIMENSION];//input vector

	/*De Jong's function*/	
	/*
	for(int i=0;i<DIMENSION;i++){
		upper[i] = 5.12;
		lower[i] = -5.12;
	}
	
	Decoder_real(indi,upper,lower,x);
	for(int i=0;i<DIMENSION;i++){
		result += x[i]*x[i];
	}*/

	/*Rastrigin's function
	for(int i=0;i<DIMENSION;i++){
		upper[i] = 5.12;
		lower[i] = -5.12;
	}
	
	Decoder_real(indi,upper,lower,x);
	for(int i=0;i<DIMENSION;i++){
		result += x[i]*x[i]-10*cos(2*PI*x[i]);
	}
	result+=10*DIMENSION;*/

	/*Schwefel's function
	for(int i=0;i<DIMENSION;i++){
		upper[i] = 500;
		lower[i] = -500;
	}
	
	Decoder_real(indi,upper,lower,x);
	
	for(int i=0;i<DIMENSION;i++){
		result += x[i]*sin(pow(abs(x[i]),0.5));
	}
	
	result = 418.9829*DIMENSION-result;*/

	/*Rosenbrock's valley*/
	for(int i=0;i<DIMENSION;i++){
		upper[i] = 2.048;
		lower[i] = -2.048;
	}
	
	Decoder_real(indi,upper,lower,x);

	for(int i=0;i<DIMENSION-1;i++){
		result += 100*pow((x[i]*x[i]-x[i+1]),2.0) + pow((1-x[i]),2.0);
	}
	
	/*Griewangk's function
	for(int i=0;i<DIMENSION;i++){
		upper[i] = 600;
		lower[i] = -600;
	}
	
	Decoder_real(indi,upper,lower,x);

	double tmp_sig=0;
	for(int i=0;i<DIMENSION;i++){
		tmp_sig += x[i]*x[i];
	}

	double tmp_pi=1;
	for(int i=0;i<DIMENSION;i++){
		tmp_pi *= cos(x[i]/pow(i+1,0.5));
	}

	result = tmp_sig/4000 - tmp_pi +1;*/

	/*Ackley's function
	int		_a = 20;
	double	_b = 0.2;
	double	_c = 2*PI;

	for(int i=0;i<DIMENSION;i++){
		upper[i] = 32.768;
		lower[i] = -32.768;
	}
	
	Decoder_real(indi,upper,lower,x);

	double tmp_sig1 = 0;
	double tmp_sig2 = 0;

	for(int i=0;i<DIMENSION;i++){
		tmp_sig1 += x[i]*x[i];
	}
	tmp_sig1 /= DIMENSION;

	for(int i=0;i<DIMENSION;i++){
		tmp_sig2 += cos(_c*x[i]);
	}
	tmp_sig2 /= DIMENSION;

	result = -(_a) * exp(-(_b) * pow(tmp_sig1,0.5)) - exp(tmp_sig2) + _a + exp(1);*/

	return result;
}
__host__ __device__	void	Decoder_real			(int* indi,double* upper,double* lower,double* vec)
{
	/*
		Description:
		Decoder for dna individual in double type

		Input:
		target dna individual,
		upper bound,
		lower bound,
		result vector.
	*/	
	int	Trasl_table[4][4][4]={ {{1,1,2,2},{3,3,3,3},{4,4,0,0},{5,5,0,6}} ,
							   {{2,2,2,2},{7,7,7,7},{8,8,9,9},{10,10,10,10}} ,
							   {{11,11,12,12},{13,13,13,13},{14,14,15,15},{3,3,10,10}} ,
							   {{16,16,16,16},{17,17,17,17},{18,18,19,19},{20,20,20,20}}};

	
	double* REAL_UP_BOUND = (double*) malloc(sizeof(double) * DIMENSION);
	double* REAL_LOW_BOUND = (double*) malloc(sizeof(double) * DIMENSION);	

	for(int i=0;i<DIMENSION;i++){
		REAL_UP_BOUND[i] = upper[i];
		REAL_LOW_BOUND[i] = lower[i];
	}

	int	amino_acid=0;
	int	amino_range=_AMINO_RANGE;
	
	for(int d=0;d<DIMENSION;d++){//for each dimension

		int codon[CODON_LENGTH]={};
		for(int c=0;c<CODON_LENGTH;c++){//get codon base
			codon[c] = *(indi+c+d*CODON_LENGTH);
		}		

		amino_acid=0;
		for(int i=0;i<CODON_DIGIT;i++){//for each digit
			int index = 0+i*3;
			amino_acid+=Trasl_table[codon[index]][codon[index+1]][codon[index+2]] *pow(21.0,i);
		}

		*(vec+d) = amino_acid/(double)amino_range*(REAL_UP_BOUND[d]-REAL_LOW_BOUND[d])+REAL_LOW_BOUND[d];
	}

	free(REAL_UP_BOUND);
	free(REAL_LOW_BOUND);
}


__host__ __device__	void	Bobble_Sort				(double* target,int* ind,int size)
{
	/*
		Description:
		Bobble sorting.Sort target array and its index from less to large.

		Input:
		unsorted target array,
		index for unsorted target array,
		size of array.
	*/
	double	tmp;
	int		ind_tmp;

	for(int i=0;i<size;i++)
		ind[i] = i;

	for(int i=0;i<size-1;i++){
		for(int j=0;j<size-i-1;j++){
			if(target[j] > target[j+1]){
				tmp = target[j+1];
				target[j+1] = target[j];
				target[j] = tmp;

				ind_tmp = ind[j+1];
				ind[j+1] = ind[j];
				ind[j] = ind_tmp;
			}
		}
	}
}
__host__ __device__	void	Bobble_Sort_inv			(double* target,int* ind,int size)
{
	/*
		Description:
		Bobble sorting.Sort target array and its index from large to less.

		Input:
		unsorted target array,
		index for unsorted target array,
		size of array.
	*/
	double	tmp;
	int		ind_tmp;

	for(int i=0;i<size;i++)
		ind[i] = i;

	for(int i=0;i<size-1;i++){
		for(int j=0;j<size-i-1;j++){
			if(target[j] < target[j+1]){
				tmp = target[j+1];
				target[j+1] = target[j];
				target[j] = tmp;

				ind_tmp = ind[j+1];
				ind[j+1] = ind[j];
				ind[j] = ind_tmp;
			}
		}
	}
}

__host__ __device__	double	Evaluation				(int* indi,bool opt)
{	
	/*
		Description:
		Evaluate input individual.		

		Input:
		input individual.
		optimal selection.

		Output:
		fitness.
	*/	
	double	fit = TestProblem(indi);
	if(opt){//find min
		return	-fit;
	}else{//find max
		return	fit;
	}
}
__host__ __device__	void	Schema_Normalize		(double* Schema)
{
	/*
		Description:
		Normalize input schema.

		Input:
		Input schema
	*/
	for(int j=0;j<FIT_LENGTH;j++){
		double tmp_sum=0;
		for(int i=0;i<4;i++)
			tmp_sum+=Schema[i*FIT_LENGTH+j];
		if(tmp_sum!=0){
			for(int i=0;i<4;i++)
				Schema[i*FIT_LENGTH+j]/=tmp_sum;
		}
	}
}
__host__ __device__	double	Find_max				(double* ary,int size,int& index)
{
	/*
		Description:
		Find max value of input array in double type

		Input:
		target array,
		size of target array.

		Output:
		max value of input array.
	*/
	double temp = *ary;
	index = 0;
	for(int i=1;i<size;i++){
		if(*(ary+i) > temp){
			temp = *(ary+i);
			index = i;
		}			
	}
	return temp;
}
__host__ __device__	double	Find_max				(double* ary,int size)
{
	/*
		Description:
		Find max value of input array in double type

		Input:
		target array,
		size of target array.

		Output:
		max value of input array.
	*/
	double temp = *ary;
	for(int i=1;i<size;i++){
		if(*(ary+i) > temp)
			temp = *(ary+i);
	}
	return temp;
}
__host__ __device__	double	Find_min				(double* ary,int size)
{
	/*
		Description:
		Find max value of input array in double type

		Input:
		target array,
		size of target array.

		Output:
		max value of input array.
	*/
	double temp = *ary;
	for(int i=1;i<size;i++){
		if(*(ary+i) < temp)
			temp = *(ary+i);
	}
	return temp;
}
__host__ __device__	int		Schema_base				(double* interval,int col,int randnum,double rand_bit)
{
	/*
		Description:
		Find base for target bit according to schema

		Input: 
		interval for target bit, 
		target bit, 		
		random number for base, 
		random number for probability.

		Output:
		base for target bit
	*/
	int next_base;

	double	T_intrvl_begin = 0							,T_intrvl_end = interval[0*FIT_LENGTH+col];
	double	C_intrvl_begin = interval[0*FIT_LENGTH+col]	,C_intrvl_end = interval[1*FIT_LENGTH+col];
	double	A_intrvl_begin = interval[1*FIT_LENGTH+col]	,A_intrvl_end = interval[2*FIT_LENGTH+col];
	double	G_intrvl_begin = interval[2*FIT_LENGTH+col]	,G_intrvl_end = interval[3*FIT_LENGTH+col];						

	if(rand_bit>T_intrvl_begin && rand_bit<=T_intrvl_end)
		next_base = 0;
	else if(rand_bit>C_intrvl_begin && rand_bit<=C_intrvl_end)
		next_base = 1;
	else if(rand_bit>A_intrvl_begin && rand_bit<=A_intrvl_end)
		next_base = 2;
	else if(rand_bit>G_intrvl_begin && rand_bit<=G_intrvl_end)
		next_base = 3;
	else
		next_base = randnum;
	
	return next_base;
}
__host__ __device__	void	Interval				(double* schema,double* interval)
{
	/*
		Description:
		Get schema probability interval.

		Input:
		input schema.
		schema probability interval.
	*/

	//Get probability interval	
	for(int j=0;j<FIT_LENGTH;j++){
		double tmp=0;
		for(int i=0;i<4;i++){
			tmp += *(schema+i*FIT_LENGTH+j);
			*(interval+i*FIT_LENGTH+j) = tmp;
		}
	}
	
}
__host__ __device__	double	GetEntropy				(double* data,int data_size)
{
	//Calculate Entropy
	double	temp_H=0;
	for(int i=0;i<data_size;i++){
		double p = *(data+i);
		if(p==0){
			temp_H += 0;
		}else{
			temp_H += -p*log2(p);
		}
	}

	return	temp_H;
}
#endif


