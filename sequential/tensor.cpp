#include "tensor.h"
#include <stdlib.h>
#include <time.h>

Tensor::Tensor(int h,int w,int dimension,std::string n){
	height = h;
	width = w;
	dim = dimension;
	name = n;
	srand(time(NULL));
	data = new int[h*w*dimension];
}

Tensor::Tensor(int h,int w,int dimension,double* d,std::string n){
	height = h;
	width = w;
	dim = dimension;
	data = d;
	name = n;
}

double* Tensor::get_data(){
	return data;
}

std::string Tensor::get_name(){
	return name;
}