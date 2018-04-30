#include "tensor.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>

Tensor::Tensor() {}

Tensor::Tensor(int h,int w,int dimension,int f, std::string n){
	height = h;
	width = w;
	dim = dimension;
	name = n;
	srand(time(NULL));
	num_filter = f;
	data = new double[h*w*dimension];
}

Tensor::Tensor(int h,int w,int dimension,int f, double* d,std::string n){
	height = h;
	width = w;
	dim = dimension;
	data = d;
	name = n;
	num_filter = f;
}

//Tensor::~Tensor(){
//	delete[] data;
//}

void Tensor::print_t(){
	for(int i=0;i<height*width*dim*num_filter;i++)
	{
		std::cout<<data[i]<<' ';
	}
	std::cout<<'\n';
}

double* Tensor::get_data(){
	return data;
}

std::string Tensor::get_name(){
	return name;
}
