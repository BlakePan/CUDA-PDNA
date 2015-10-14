#include	"Config.h"
#include	"mykernel.cuh"

void	CUDASetDevice();
void	OutputSpec(const cudaDeviceProp);
FILE*	Open_file(string file_name,string file_name_exten);
void	inline	randomize()//random set up
{
	time_t t;
	srand((unsigned) time(&t));	
}
void	cudasafe( cudaError_t error, char* message)
{
   if(error!=cudaSuccess) { 
	   fprintf(stderr,"ERROR: %s : %i\n",message,error); 
	   const char *E = cudaGetErrorString(cudaGetLastError());
	   printf("%s\n", E);
	   system("pause");
	   exit(-1); 
   }
}

//Dimension
const	dim3	GRID(TUBE,1);							//dimension of grid
const	dim3	BLOCK(POPULATION,1);					//dimendion of block

int		main()
{
	//=====Set Up=====//
	CUDASetDevice();	//select device for cuda
	randomize();		//set up random for host
	string	E;			//error message		

	//=====Alocate Memory=====//
	//CUDA curand	
	curandState* devStates;												
	size_t	size_state		= (POPULATION);								
	size_t	mem_size_state	= sizeof(curandState) * TUBE * size_state;	
	cudaMalloc((void **)&devStates, mem_size_state);cudasafe(cudaGetLastError(),"Allocate curand");	

	dim3 D_curand(size_state,1);												
	/*setup_kernel<<<GRID,D>>>(devStates,rand());							
	cudaDeviceSynchronize();
	cudasafe(cudaGetLastError(),"set up curand");	*/

	//Populations	
	int*	d_popu;														
	size_t	size_popu		= (POPULATION * FIT_LENGTH * TUBE);			
	size_t	mem_size_popu	= sizeof(int) * size_popu;	
	cudaMalloc((void **)&d_popu, mem_size_popu);cudasafe(cudaGetLastError(),"Allocate Populations");
	cudaMemset(d_popu,0,mem_size_popu);					

	//Scores
	double*	h_score;													
	double*	d_score;													
	size_t	size_score		= (POPULATION * TUBE);						
	size_t	mem_size_score	= sizeof(double) * size_popu;				
	h_score = (double*)malloc(mem_size_score);							
	cudaMalloc((void **)&d_score, mem_size_score);cudasafe(cudaGetLastError(),"Allocate Scores");
	cudaMemset(d_score,0,mem_size_score);			

	//Fittness	
	double*	d_fit;														
	size_t	size_fit		= (POPULATION * TUBE);						
	size_t	mem_size_fit	= sizeof(double) * size_fit;					
	cudaMalloc((void **)&d_fit, mem_size_fit);cudasafe(cudaGetLastError(),"Allocate Fittness");							
	cudaMemset(d_fit,0,mem_size_fit);	

	//Recording
	double*	h_rec;
	double*	d_rec;
	size_t	size_rec = (ITER * TUBE);
	size_t	mem_size_rec = sizeof(double) * size_rec;
	h_rec = (double*)malloc(mem_size_rec);
	cudaMalloc((void **)&d_rec, mem_size_rec);cudasafe(cudaGetLastError(),"Allocate Record");
	cudaMemset(d_rec,0,mem_size_rec);	

	//Sample for Schema 	
	int*	d_smp_schema;												
	size_t	size_smp_schema		= (TUBE * FIT_LENGTH);	
	size_t	mem_size_smp_schema	= sizeof(int) * size_smp_schema;	
	cudaMalloc((void **)&d_smp_schema, mem_size_smp_schema);cudasafe(cudaGetLastError(),"Allocate Sample for Schema");
	cudaMemset(d_smp_schema,0,mem_size_smp_schema);	

	//Sample for Enzyme	
	int*	d_smp_enzyme;
	size_t	size_smp_enzyme		= (TUBE * FIT_LENGTH);
	size_t	mem_size_smp_enzyme	= sizeof(int) * size_smp_enzyme;	
	cudaMalloc((void **)&d_smp_enzyme, mem_size_smp_enzyme);cudasafe(cudaGetLastError(),"Allocate Sample for Enzyme");
	cudaMemset(d_smp_enzyme,0,mem_size_smp_enzyme);	

	//Restriction Enzyme 	
	double*	d_enzyme;
	size_t	size_enzyme			= (4 * FIT_LENGTH);
	size_t	mem_size_enzyme		= sizeof(double) * size_enzyme;
	cudaMalloc((void **)&d_enzyme, mem_size_enzyme);cudasafe(cudaGetLastError(),"Allocate Restriction Enzyme");
	cudaMemset(d_enzyme,0,mem_size_enzyme);		

	//Restriction Enzyme Vector
	int*	d_enzyme_vec;
	size_t	size_enzyme_vec		= (FIT_LENGTH);
	size_t	mem_size_enzyme_vec = sizeof(int) * size_enzyme_vec;
	cudaMalloc((void **)&d_enzyme_vec, mem_size_enzyme_vec);cudasafe(cudaGetLastError(),"Allocate Restriction Enzyme vector");
	cudaMemset(d_enzyme_vec,0,mem_size_enzyme_vec);

	//Global schema		
	double* d_g_schema;
	size_t	mem_size_schema = sizeof(double) * 4 * FIT_LENGTH;
	cudaMalloc((void**) &d_g_schema,mem_size_schema);cudasafe(cudaGetLastError(),"Allocate memory for schema");
	cudaMemset(d_g_schema,0,mem_size_schema);

	//Local schema		
	double* d_l_schema;	
	size_t	mem_size_local_schema = sizeof(double) * 4 * FIT_LENGTH *TUBE;
	cudaMalloc((void**) &d_l_schema,mem_size_local_schema);cudasafe(cudaGetLastError(),"Allocate memory for schema");
	cudaMemset(d_l_schema,0,mem_size_local_schema);

	//Search schema		
	double* d_s_schema;	
	size_t	mem_size_search_schema = sizeof(double) * 4 * FIT_LENGTH *TUBE;
	cudaMalloc((void**) &d_s_schema,mem_size_search_schema);cudasafe(cudaGetLastError(),"Allocate memory for schema");
	cudaMemset(d_s_schema,0,mem_size_search_schema);

	//Entropy	
	double*	d_entropy;
	size_t	mem_size_entropy = sizeof(double) * TUBE;
	cudaMalloc((void**) &d_entropy,mem_size_entropy);cudasafe(cudaGetLastError(),"Allocate memory for Entropy");
	cudaMemset(d_entropy,0,mem_size_entropy);

	//Open file	
	string file_ext_xls(" .xls");

	string file("EXL/CUDA DNA ");	
	FILE* f1 = Open_file(file,file_ext_xls);
	
	//=====Start Iteration=====//
	for(int simulat_loop=0;simulat_loop<SIMULATION;simulat_loop++){
		//cout<<simulat_loop<<endl;

		//=====Initialization=====//
		//=====Setup curand Kernel & Memory reset=====//
		setup_kernel<<<GRID,D_curand>>>(devStates,rand());
		cudaMemset(d_popu,0,mem_size_popu);								
		cudaMemset(d_score,0,mem_size_score);		
		cudaMemset(d_fit,0,mem_size_fit);
		cudaMemset(d_rec,0,mem_size_rec);
		cudaMemset(d_smp_schema,0,mem_size_smp_schema);	
		cudaMemset(d_smp_enzyme,0,mem_size_smp_enzyme);
		cudaMemset(d_enzyme,0,mem_size_enzyme);
		cudaMemset(d_enzyme_vec,0,mem_size_enzyme_vec);
		cudaMemset(d_g_schema,0,mem_size_schema);
		cudaMemset(d_l_schema,0,mem_size_local_schema);
		cudaMemset(d_s_schema,0,mem_size_search_schema);
		cudaMemset(d_entropy,0,mem_size_entropy);
		cudasafe(cudaGetLastError(),"Setup curand Kernel & Memory reset");

		//=====Initial Population Kernel=====//
		Init_Popu_kernel<<<GRID,BLOCK>>>(d_popu,devStates);cudasafe(cudaGetLastError(),"Initial Population Kernel");
		cudaDeviceSynchronize();

		//=====Initial Enzyme Kernel=====//
		Init_Enzyme_kernel<<<1,1>>>(d_enzyme);cudasafe(cudaGetLastError(),"Initial Enzyme Kernel");		
		
		cout<<endl;
		for(int I=0;I<ITER;I++){
			printf("\r%05.3f%%",(float)I/ITER*100);

			//=====Evaluation=====//
			Evaluation_kernel<<<GRID,BLOCK>>>(d_popu,d_score,d_fit);cudasafe(cudaGetLastError(),"Evaluation Kernel");
			cudaDeviceSynchronize();

			Popu_sort_kernel<<<1,GRID>>>(d_popu,d_score,d_fit);cudasafe(cudaGetLastError(),"Population sorting Kernel");			
			Record_kernel<<<1,GRID>>>(d_score,d_rec,I);cudasafe(cudaGetLastError(),"Record Kernel");

			//=====Extraction=====//
			Extract_kernel<<<1,GRID>>>(d_popu,d_smp_schema);cudasafe(cudaGetLastError(),"Extract Kernel");

			//=====Updating Restriction Enzyme=====//			
			if(I==0){
				First_Filter_kernel<<<1,GRID>>>(d_popu,d_smp_enzyme);
			}else{
				Filter_kernel<<<1,1>>>(d_popu,d_smp_enzyme,OPT);
			}						
			cudasafe(cudaGetLastError(),"Filter Kernel");

			Update_Enzyme_kernel<<<1,1>>>(d_smp_enzyme,d_enzyme,d_enzyme_vec);cudasafe(cudaGetLastError(),"Update Enzyme Kernel");			

			//=====Updating Global Schema=====//
			Update_glo_schema_kernel<<<1,1>>>(d_smp_schema,d_g_schema,d_enzyme_vec);cudasafe(cudaGetLastError(),"Update Global Schema");

			//=====Updating Local Schema=====//
			Update_loc_schema_kernel<<<1,1>>>(d_popu,d_l_schema,d_fit,d_enzyme_vec);cudasafe(cudaGetLastError(),"Update Local Schema");
			
			//=====Updating Population=====//
			Update_population<<<GRID,BLOCK>>>(d_popu,d_g_schema,d_l_schema,d_s_schema,devStates);cudasafe(cudaGetLastError(),"Update Population");
			cudaDeviceSynchronize();

			Evaluation_kernel<<<GRID,BLOCK>>>(d_popu,d_score,d_fit);cudasafe(cudaGetLastError(),"Evaluation Kernel");
			cudaDeviceSynchronize();

			Popu_sort_kernel<<<1,GRID>>>(d_popu,d_score,d_fit);cudasafe(cudaGetLastError(),"Population sorting Kernel");

			//=====Migration=====//			
			Get_Entropy<<<GRID,FIT_LENGTH>>>(d_s_schema,d_entropy);cudasafe(cudaGetLastError(),"Get Entropy");
			//cudaDeviceSynchronize();

			Migration<<<GRID,BLOCK>>>(d_entropy,d_popu,devStates);cudasafe(cudaGetLastError(),"Migration");
			cudaDeviceSynchronize();

		}//end of iteration loop
		//Print Result
		cudaMemcpy(h_rec, d_rec, mem_size_rec, cudaMemcpyDeviceToHost);
		cudasafe(cudaGetLastError(),"cuda memory copy:rec");
				
		for(int i=0;i<ITER;i++){
			double	tmp[TUBE] = {};
			for(int j=0;j<TUBE;j++){		
				tmp[j]= *(h_rec+i*TUBE+j);
			}			
			
			if(OPT){
				fprintf(f1,"%.40f\t",Find_min(tmp,TUBE));
			}else{
				fprintf(f1,"%.40f\t",Find_max(tmp,TUBE));
			}
			
		}	
		fprintf(f1,"\n");
	}//end of simulation loop

	fclose(f1);

	//=====Release Memory=====//
	cudaFree(devStates);
	cudaFree(d_popu);
	cudaFree(d_score);
	cudaFree(d_fit);
	cudaFree(d_rec);
	cudaFree(d_smp_schema);
	cudaFree(d_smp_enzyme);	
	cudaFree(d_enzyme);
	cudaFree(d_enzyme_vec);
	cudaFree(d_g_schema);
	cudaFree(d_l_schema);
	cudaFree(d_s_schema);
	cudaFree(d_entropy);
	free(h_score);
	free(h_rec);
	return 0;
}
void	CUDASetDevice()
{
	//Get number of devices
	int count;
	cudaGetDeviceCount(&count);
	//Get device information
	 
	for( int i = 0; i < count; ++ i ){
		printf( "\n=== Device %i ===\n", i );
		cudaDeviceProp  sDeviceProp;
		cudaGetDeviceProperties( &sDeviceProp, i );
		OutputSpec( sDeviceProp );
	}
	cout<<endl;

	//Set device
	int dev;
	cout<<"Select Device:";
	cin>>dev;
	cudaSetDevice(dev);
	//cout<<i<<endl;
}
void	OutputSpec( const cudaDeviceProp sDevProp )
{
  printf( "Device name: %s\n", sDevProp.name );
  printf( "Device memory: %d\n", sDevProp.totalGlobalMem );
  printf( " Memory per-block: %d\n", sDevProp.sharedMemPerBlock );
  printf( " Register per-block: %d\n", sDevProp.regsPerBlock );
  printf( " Warp size: %d\n", sDevProp.warpSize );
  printf( " Memory pitch: %d\n", sDevProp.memPitch );
  printf( " Constant Memory: %d\n", sDevProp.totalConstMem );
  printf( "Max thread per-block: %d\n", sDevProp.maxThreadsPerBlock );
  printf( "Max thread dim: ( %d, %d, %d )\n", sDevProp.maxThreadsDim[0], sDevProp.maxThreadsDim[1], sDevProp.maxThreadsDim[2] );
  printf( "Max grid size: ( %d, %d, %d )\n", sDevProp.maxGridSize[0], sDevProp.maxGridSize[1], sDevProp.maxGridSize[2] );
  printf( "Ver: %d.%d\n", sDevProp.major, sDevProp.minor );
  printf( "Clock: %d\n", sDevProp.clockRate );
  printf( "textureAlignment: %d\n", sDevProp.textureAlignment );
  printf( "kernelExecTimeoutEnabled: %d\n", sDevProp.kernelExecTimeoutEnabled );
}

FILE*	Open_file(string file,string file_ext)
{
	tm* ptrnow;
	time_t loc_now = 0;
	time(&loc_now);
	ptrnow = localtime(&loc_now);
	string runtime = asctime(ptrnow);

	while(1){
		size_t found = runtime.find(":");
		if(found>=0 && found<runtime.size()){
			runtime[found] = '-';
		}else{
			break;
		}
	}
	
	string runtime2;
	for(int i=0;i<runtime.size()-1;i++){
		runtime2+=runtime[i];
	}

	file = file+runtime2+file_ext;
	FILE* f1 = fopen(file.c_str(), "w+");
	if(f1 == NULL){
		printf("error open %s file to write\n",file.c_str());
		system("pause");
		exit(1);
	}
	return f1;
}
