#ifndef	_CONFIG_H_
#define	_CONFIG_H_

#define	MIN				(true)
#define	MAX				(false)
#define	PI				(3.14159)	
#define	ROUND(X)		(int)((X)+0.5)
#define	FIND_MAX(a,b)	((a)>(b)?(a):(b))

//Problem Define
#define	DIMENSION		(2)

//Designer Define
#define	CODON_LENGTH	(6)
#define	CODON_DIGIT		(CODON_LENGTH/3)
#define	FIT_LENGTH		(CODON_LENGTH*DIMENSION)
#define	OPT				(MIN)
#define	TUBE			(10)
#define	ITER			(100)
#define	POPULATION		(50)
#define	SIMULATION		(1)

#define	COMB_GLO		(0.1)
#define	COMB_LOC		(1-COMB_GLO)

#define	POPU_SCHEMA		((int)(POPULATION*0.90))
#define	POPU_ELITE		(1)
#define	POPU_RAND		(POPULATION-POPU_SCHEMA-POPU_ELITE)

#define	MAX_H			((FIT_LENGTH-DIMENSION)*2)
#define	MAX_DIV			((FIT_LENGTH)*2)
#define	MIGRAT_TH		((double)(MAX_DIV*0.2))
#define	MIGRAT_ERR_TOL	(1)
#define	ENZ_MASK_TH		(0.5)//higher the th,more don't care

__device__	double _AMINO_RANGE=0;
#endif

#ifndef	_MY_INCLUDE_
#define	_MY_INCLUDE_
#include	"cuda_runtime.h"
#include	"device_launch_parameters.h"

#include	<stdio.h>
#include	<cuda.h>
#include	<iostream>
#include	<time.h>
#include	<cstdio>
#include	<curand_kernel.h>
#include	<bitset>
#include	<string>
#include	<iomanip>
#include	<fstream> 

using namespace std;
#endif