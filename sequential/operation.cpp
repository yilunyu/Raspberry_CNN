#include "operation.h"
#include <cmath>
#include <iostream>


Operation::Operation(std::vector<Tensor> & tens,
                      std::string op_name,
                      std::string out_name){
  name = op_name;
  inputs = tens;
  Tensor original = inputs.at(0);
  Tensor t(original.height,original.width,original.dim,1,out_name);
  output = t;
}

Operation::Operation(){
}

Tensor Operation::get_output(){
  return output;
}



double convolve_1d(double* original_start,double* filter_start,int original_width,int filter_len){
  double accum=0;
  for(int i=0;i<filter_len;i++){
    for(int j=0;j<filter_len;j++){
      accum+=*(original_start+i*original_width+j)*(*(filter_start+i*filter_len+j));
    }
  }
  return accum;
}

// Convolution::Convolution
Convolution::Convolution(std::vector<Tensor> & tens,
        std::string op_name,
        std::string out_name):Operation(){
  name = op_name;
  inputs = tens;
  Tensor original = inputs.at(0);
  Tensor weights = inputs.at(1);
  int w_bound = original.width-weights.width+1;
  int h_bound = original.height-weights.height+1;
  Tensor t(h_bound,w_bound,weights.num_filter,1,out_name);
  output = t;
}

//supports "VALID", stride is 1
void Convolution::apply_function(){
  //std::cout<< "dims "<<num_filters<<' '<<h_bound<<' '<<w_bound<<'\n';
  //std::cout<< "output "<<output.num_filter<<' '<<output.height<<' '<<output.width<<' '<<output.dim<<'\n';

  //Tensor bias = inputs.at(2);
  Tensor weights = inputs.at(1);
  Tensor original = inputs.at(0);

  //double* bias_data = bias.get_data();
  double* ori_data = original.get_data();
  double* weight_data = weights.get_data();
  double* out_data = output.get_data();

  int filter_start = 0;
  int w_bound = original.width-weights.width+1;
  int h_bound = original.height-weights.height+1;
  int num_filters = weights.num_filter;

  for(int i=0;i<num_filters;i++){
    for(int j=0;j<h_bound;j++){
      for(int k=0;k<w_bound;k++){
        out_data[i*h_bound*w_bound+j*w_bound+k] = 0.;
      }
    }
  }

  for(int i=0;i<num_filters;i++){
    int d_ori_start = 0;
    for(int j=0;j<original.dim;j++){
      for(int k=0;k<h_bound;k++){
        for(int l=0;l<w_bound;l++){
          out_data[i*h_bound*w_bound+k*w_bound+l]+=
          convolve_1d(&ori_data[d_ori_start+k*original.width+l],
                      &weight_data[filter_start+j*weights.height*weights.width],
                     original.width,
                      weights.width);
          //std::cout << k <<' '<<l<<'\n';
          //std::cout<<out_data[i*h_bound*w_bound+k*w_bound+l]<<
          //    ' '<<i*h_bound*w_bound+k*w_bound+l<<' '<<ans<<'\n';
        }
        //out_data[i*h_bound*w_bound+j*w_bound+k]+=bias_data[i];
      }
      d_ori_start+=original.width*original.height;
    }
    filter_start+=weights.width*weights.height*weights.dim;
  }

  return;
}

//input tensor is 1D
FC::FC(std::vector<Tensor> & tens,
        std::string op_name,
        std::string out_name):Operation(){
  name = op_name;
  inputs = tens;
  Tensor w = inputs.at(1);
  int weight_h = w.height;
  //std::string w_name = op_name;
  //w_name.append("_w");
  //Tensor w(num_output,weight_h,1,1,w_name);
  //std::string b_name = op_name;
  //w_name.append("_b");
  //Tensor b(num_output,1,1,1,b_name);

  //initialize b and w

  Tensor t(1,weight_h,1,1,out_name);
  output = t;
}

void FC::mul(){
  Tensor weights = inputs.at(1);
  Tensor original = inputs.at(0);
  double* ori_data = original.get_data();
  double* weight_data = weights.get_data();
  for(int j=0;j<original.height;j++){
    for(int k=0;k<weights.width;k++){
      int ori_dim_start = 0;
      int weight_dim_start = 0;
      double accum = 0;
      for(int l=0;l<original.dim;l++){
        int ori_start = ori_dim_start+j*original.width;
        int weight_start = weight_dim_start+k;
        for(int i=0;i<original.width;i++){
          accum+=ori_data[ori_start]*weight_data[weight_start];
          ori_start++;
          weight_start+=weights.width;
        }
//        out_data[]
        ori_dim_start+=original.width*original.height;
        weight_dim_start+=weights.width*weights.height;
      }
    }
  }

  return;
}

void FC::apply_function(){
  Tensor bias = inputs.at(2);
  Tensor weights = inputs.at(1);
  Tensor original = inputs.at(0);
  double* bias_data = bias.get_data();
  double* ori_data = original.get_data();
  double* weight_data = weights.get_data();
  double* out_data = output.get_data();
  int weight_start = 0;
  for(int i=0;i<weights.height;i++){
    double accum=0;
    for(int j=0;j<original.height*original.width*original.dim;j++){
      accum+=ori_data[j]*weight_data[j+weight_start];
    }
    out_data[i] = accum+bias_data[i];
    weight_start+=original.height*original.width*original.dim;
  }

  return;
}

//2 by 2 kernel max pooling
Pooling::Pooling(std::vector<Tensor> & original,std::string op_name,
		std::string out_name):Operation(){		
  name = op_name;
  inputs = original;
  Tensor tmp = original.at(0);
  int w_bound = tmp.width/2;
  int h_bound = tmp.height/2;
  Tensor t(h_bound,w_bound,tmp.dim,1,out_name);
  output = t;
}

//supports "VALID" right now
void Pooling::apply_function()
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
      for(int k=0;k<w_bound;k+=2){
	std::cout<<j<<' '<<k<<'\n';
        int cur_max = -9999;
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

Flatten::Flatten(std::vector<Tensor> & tens,std::string op_name,
        std::string out_name):Operation(){
  name = op_name;
  inputs = tens;
  Tensor original = inputs.at(0);
  Tensor t(original.height*original.width*original.dim,1,1,1,out_name);
  output = t;

}

void Flatten::apply_function()
{
  Tensor original = inputs.at(0);
  double* output_data = output.get_data();
  double* ori_data = original.get_data();
  for(int i=0;i<output.width;i++)
  {
    output_data[i] = ori_data[i];
  } 
}

Relu::Relu(Tensor original,std::string op_name,std::string out_name):Operation()
{
  inputs.push_back(original);
  name = op_name;
  Tensor t(original.height,original.width,original.dim,1,out_name);
  output = t;
}
/*
Relu::Relu(std::vector<Tensor> original,std::string op_name,std::string out_name):Operation()
{
  inputs = original;
  name = op_name;
  original = original.at(0);
  Tensor t(original.height,original.width,original.dim,1,out_name);
  output = t;
}
*/
void Relu::apply_function()
{
  Tensor original = inputs.at(0);
  double *ori_data = original.get_data(),
  *out_data = output.get_data();
  for(int i=0;i<original.width*original.height*original.dim;i++){
    if(ori_data[i]<0){
      out_data[i] = 0;
    }
    else{
      out_data[i] = ori_data[i];
    }
  }
}

Softmax::Softmax(Tensor original,std::string op_name,std::string out_name):Operation()
{
  inputs.push_back(original);
  name = op_name;
  Tensor t(original.height,original.width,original.dim,1,out_name);
  output = t;
}
/*
Softmax::Softmax(std::vector<Tensor> original,std::string op_name,std::string out_name):Operation()
{
  inputs = original;
  name = op_name;
  original = original.at(0);
  Tensor t(original.height,original.width,original.dim,1,out_name);
  output = t;
}
*/
void Softmax::apply_function()
{
  Tensor original = inputs.at(0);
  double *ori_data = original.get_data(),
  *out_data = output.get_data();
  double max = -99999;
  for(int i=0;i<original.width*original.height*original.dim;i++)
  {
    if(ori_data[i]>max)
    {
      max = ori_data[i];
    }
  }
  double sum = 0.;
  for(int i=0;i<original.width*original.height*original.dim;i++)
  {
    out_data[i] = exp(ori_data[i]-max);
    sum+=out_data[i];
  }
  for(int i=0;i<original.width*original.height*original.dim;i++)
  {
    out_data[i] /=sum;
  }
  return;
}
