#include "tensor.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>

Tensor::Tensor() {}

Tensor::Tensor(ne10_int32_t h,ne10_int32_t w,ne10_int32_t dimension,ne10_int32_t f, std::string n){
	height = h;
	width = w;
	dim = dimension;
	name = n;
	srand(time(NULL));
	num_filter = f;
	data = new ne10_float32_t[h*w*dimension];
}

Tensor::Tensor(ne10_int32_t h,ne10_int32_t w,ne10_int32_t dimension,ne10_int32_t f, ne10_float32_t
* d,std::string n){
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
	for(ne10_int32_t i=0;i<height*width*dim*num_filter;i++)
	{
		std::cout<<data[i]<<' ';
	}
	std::cout<<'\n';
}

ne10_float32_t
* Tensor::get_data(){
	return data;
}

std::string Tensor::get_name(){
	return name;
}
