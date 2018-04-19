#include "operation.h"
#include <cmath>
double* Operation::get_output(){
	return output;
}

double* Operation::apply_function(){
	return;
}

Operation::Operation(std::vector<Tensor> in,std::string op_name,std::string out_name)
{
	name = op_name;
	inputs = in;
	Tensor original = in.at(0);
	Tensor t(original.height,original.width,original.dim,1,out_name);
	output = t;
}

int convolve_1d(double* original_start,double* filter_start,int original_width,){

}

//supports "VALID", kernels are 5x5
void Convolution::apply_function(){
	Tensor bias = inputs.at(2);
	Tensor weights = inputs.at(1);
	Tensor original = inputs.at(0);

	double* bias_data = bias.get_data();
	double* ori_data = original.get_data();
	double* weight_data = weight.get_data();
	double* out_data = output.get_data();

	int filter_start = 0;
	int w_bound = original.width-5+1;
	int h_bound = original.height-5+1;
	for(int i=0;i<num_filters;i++){
		int d_start = 0;
		for(int j=0;j<original.dim;j++){
			for(int k=0;k<h_bound;k++){
				for(int l=0;l<w_bound;l++){



				}
			}
			
		}
		filter_start+=weights.width*weights.height*weights.dim;
	}
	return;
}

void FC::mul(){
	Tensor weights = inputs.at(1);
	Tensor original = inputs.at(0);
	double* ori_data = original.get_data();
	double* weight_data = weight.get_data();
	double* out_data = output.get_data();
	for(int j=0;j<original.height;j++){
		for(int k=0;k<weights.width;k++){
			int ori_dim_start = 0;
			int weight_dim_start = 0;
			double accum = 0;
			for(int l=0;l<original.dim;l++){
				int ori_start = ori_dim_start+j*original.width;
				int weight_start = weight_dim_start+k;
				for(int i=0;i<original.width;i++){
					accum+=ori_data[ori_start]*weight_data[weight_start];
					ori_start++;
					weight_start+=weights.width;
				}
//				out_data[]
				ori_dim_start+=original.width*original.height;
				weight_dim_start+=weights.width*weights.height;
			}
		}
	}

	return;
}

void FC::apply_function(){
	Tensor bias = inputs.at(2);
	Tensor weights = inputs.at(1);
	Tensor original = inputs.at(0);

	double* bias_data = bias.get_data();
	double* ori_data = original.get_data();
	double* weight_data = weight.get_data();
	double* out_data = output.get_data();

	int weight_start = 0;
	for(int i=0;i<weights.width;i++){
		double accum=0;
		for(int j=0;j<original.height*original.width*original.dim;j++){
			accum+=ori_data[j]*weight_data[j+weight_start];
		}		
		out_data[i] = accum+bias_data[i];
		weight_start+=original.height*original.width*original.dim;
	}

	return;
}

//supports "VALID" right now
void Pooling:::apply_function()
{
	Tensor original = inputs.at(0);
	double *ori_data = original.get_data();
	double *out_data = output.get_data();
    int ori_start = 0;
    int h_bound = original.height-original.height%2;
    int w_bound = original.width-original.width%2;
    int out_width = w_bound/2;
    int out_height = h_bound/2;
    
	for(int i=0;i<original.dim;i++){

		for(int j=0;j<h_bound;j+=2){
			for(int k=0;k<w_bound;j+=2){
				int cur_max = 0;
				if(ori_data[j*original.width+k+ori_start]>ori_data[j*original.width+k+ori_start+1])
				{
					cur_max =ori_data[j*original.width+k+ori_start]; 
				}
				else
				{
					cur_max =ori_data[j*original.width+k+ori_start+1]; 
				}
				if(ori_data[(j+1)*original.width+k+ori_start]>cur_max)
				{
					cur_max =ori_data[(j+1)*original.width+k+ori_start]; 
				}
				if(ori_data[(j+1)*original.width+k+1+ori_start]>cur_max)
				{
					cur_max =ori_data[(j+1)*original.width+k+1+ori_start]; 
				}
				out_data[i*out_width*out_height+j/2*out_width+k/2] = cur_max;
			}
		}
		ori_start+=original.height*original.width;
	}	
}

void Relu:::apply_function()
{
	Tensor original = inputs.at(0);
	double *ori_data = original.get_data();
	for(int i=0;i<original.width*original.height*original.dim;i++){
		if(ori_data[i]<0){
			output[i] = 0;
		}
		else{
			output[i] = ori_data[i];
		}
	}
}

void SoftMax:::apply_function()
{
	Tensor original = inputs.at(0);
	double *ori_data = original.get_data();
	double max = -99999;
	for(int i=0;i<original.width*original.height*original.dim;i++)
	{
		if(ori_data[i]>max)
		{
			max = ori_data;
		}
	}
	double sum = 0.;
	for(int i=0;i<original.width*original.height*original.dim;i++)
	{
		output[i] = exp(ori_data[i]-max);
		sum+=output[i];
	}
	for(int i=0;i<original.width*original.height*original.dim;i++)
	{
		output[i] /=sum;
	}
	return;
}