#include "operation.h"
#include <cmath>
#include <iostream>
#include <arm_neon.h>
#include <omp.h>

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



inline void convolve_1d(ne10_float32_t* out, ne10_float32_t* ori_data,
                            ne10_float32_t filter_val,int len){
  int j;
  int leftover = len%4;
  int limit = len-leftover;
  float32x4_t filter_val_vec = vld1q_dup_f32(&filter_val);
  for(j=0;j<limit;j+=4){
   
    float32x4_t in1 = vld1q_f32(&ori_data[j]);
    float32x4_t out_old = vld1q_f32(&out[j]);
    float32x4_t tmp = vmlaq_f32(out_old,filter_val_vec,in1);
    vst1q_f32(&out[j],tmp);
  }
  for(j=limit;j<limit+leftover;j++){
    out[j]+=ori_data[j]*filter_val;
  }
    
}

//w_len is number of multiplications needed per row
//source_width is source image's width
inline void convolve_2d(ne10_float32_t* out,ne10_float32_t* ori_start,
                         ne10_float32_t filter_val, int h_len,int w_len,int source_width)
{
  int row_start =0;
  #pragma omp parallel for
  for(int i=0;i<h_len;i++){
    convolve_1d(&out[i*w_len],&ori_start[row_start],filter_val,w_len);
    row_start+=source_width;
  }
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

  Tensor weights = inputs.at(1);
  Tensor original = inputs.at(0);

  ne10_float32_t* ori_data = original.get_data();
  ne10_float32_t* weight_data = weights.get_data();
  ne10_float32_t* out_data = output.get_data();

  int filter_start = 0;
  int w_bound = original.width-weights.width+1;
  int h_bound = original.height-weights.height+1;
  int num_filters = weights.num_filter;

  for(int i=0;i<num_filters;i++){
    for(int j=0;j<h_bound;j++){
      // #pragma omp parallel for
      for(int k=0;k<w_bound;k++){
        out_data[i*h_bound*w_bound+j*w_bound+k] = 0.;
      }
    }
  }

  for(int i=0;i<num_filters;i++){
    int d_ori_start = 0;
    for(int j=0;j<original.dim;j++){
      for(int k=0;k<weights.height;k++){
        for(int l=0;l<weights.width;l++){
          convolve_2d(&out_data[i*h_bound*w_bound],&ori_data[d_ori_start+k*original.width+l],
                      weight_data[filter_start+j*weights.height*weights.width+k*weights.width+l], 
                         h_bound,w_bound,original.width);
        }
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

  Tensor t(1,weight_h,1,1,out_name);
  output = t;
}

void FC::apply_function(){
  #ifdef __aarch64__ 
  std::cout<<'yes\n';
  assert(false);
  #endif
  Tensor bias = inputs.at(2);
  Tensor weights = inputs.at(1);
  Tensor original = inputs.at(0);
  ne10_float32_t* bias_data = bias.get_data();
  ne10_float32_t* ori_data = original.get_data();
  ne10_float32_t* weight_data = weights.get_data();
  ne10_float32_t* out_data = output.get_data();
  int weight_start = 0;
  for(int i=0;i<weights.height;i++){
    float32x4_t accumx4=vmovq_n_f32(0.0f);
    int j;
    int leftover = original.height*original.width*original.dim%4;
    int limit = original.height*original.width*original.dim-leftover;
    #pragma omp parallel for
    for(j=0;j<limit;j+=4){
      float32x4_t in1 = vld1q_f32(&ori_data[j]);
      float32x4_t in2 = vld1q_f32(&weight_data[weight_start+j]);
      accumx4 = vmlaq_f32(accumx4,in1,in2);
    }
    out_data[i] = accumx4[0]+accumx4[1]+accumx4[2]+accumx4[3]+bias_data[i];
    #pragma omp parallel for
    for(j=limit;j<limit+leftover;j++){
      out_data[i]+=ori_data[j]*weight_data[weight_start+j];
    }
    
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
  ne10_float32_t *ori_data = original.get_data();
  ne10_float32_t *out_data = output.get_data();
  int ori_start = 0;
  int h_bound = original.height-original.height%2;
  int w_bound = original.width-original.width%2;
  int out_width = w_bound/2;
  int out_height = h_bound/2;
  for(int i=0;i<original.dim;i++){

    for(int j=0;j<h_bound;j+=2){
      #pragma omp parallel for
      for(int k=0;k<w_bound;k+=2){
        float32x2_t in1 = vld1_f32(&ori_data[j*original.width+k+ori_start]);
        float32x2_t in2 = vld1_f32(&ori_data[(j+1)*original.width+k+ori_start]);
        float32x2_t m1 = vpmax_f32(in1,in2);
        float32x2_t m2 = vpmax_f32(m1,m1);
        float maxValue = vget_lane_f32(m2,0);
        out_data[i*out_width*out_height+j/2*out_width+k/2] = maxValue;
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
  ne10_float32_t* output_data = output.get_data();
  ne10_float32_t* ori_data = original.get_data();
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

void Relu::apply_function()
{
  Tensor original = inputs.at(0);
  ne10_float32_t *ori_data = original.get_data(),
  *out_data = output.get_data();

  float32x4_t mask_0 = vmovq_n_f32(0.0f);
  int leftover = original.height*original.width*original.dim%4;
  int limit = original.height*original.width*original.dim-leftover;
  int i;
  for(i=0;i<limit;i+=4){
    float32x4_t in1 = vld1q_f32(&ori_data[i]);
    uint32x4_t mask = vcltq_f32(mask_0,mask_0);
    float32x4_t v = vbslq_f32(mask,in1,mask_0);
    vst1q_f32(&out_data[i],v);
  }

  for(i=limit;i<limit+leftover;i++){
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

void Softmax::apply_function()
{
  int nthreads =omp_get_max_threads();
  Tensor original = inputs.at(0);
  ne10_float32_t *ori_data = original.get_data(),
  *out_data = output.get_data();
  ne10_float32_t maxes[nthreads];
  int i;
  int leftover = original.height*original.width*original.dim%4;
  int limit = original.height*original.width*original.dim-leftover;
  #pragma omp parallel for
  for(i=0;i<nthreads;i++)
  {
    maxes[i]= ori_data[0];
  }

  #pragma omp parallel for
  for(i=0;i<limit;i+=4)
  {
    int t = omp_get_thread_num();
    float32x4_t in1 = vld1q_f32(&ori_data[i]);
    float32x2_t max_2 = vpmax_f32(vget_low_f32(in1),vget_high_f32(in1));
    float32x2_t max_1 = vpmax_f32(max_2,max_2);
    float maxValue = vget_lane_f32(max_1,0);
    if(maxValue>maxes[t])
    {
      maxes[t] = maxValue;
    }
  }
  ne10_float32_t max=maxes[0];
  for(i=0;i<nthreads;i++){
    if(maxes[i]>max){
      max = maxes[i];
    }
  }
  for(i=limit;i<limit+leftover;i++){
    if(ori_data[i]>max){
      max = ori_data[i];
    }
  }


  ne10_float32_t sums[nthreads];
  #pragma omp parallel for
  for(i=0;i<nthreads;i++)
  {
    sums[i]= 0.0f;
  }

  #pragma omp parallel for
  for(i=0;i<original.width*original.height*original.dim;i++)
  {
    int t = omp_get_thread_num();
    out_data[i] = exp(ori_data[i]-max);
    sums[t]+=out_data[i];
  }
  ne10_float32_t sum=0.0f;
  for(i=0;i<nthreads;i++)
  {
    sum+=sums[i];
  }
  sum = 1.0f/sum;

  #pragma omp parallel for
  for(i=0;i<limit;i+=4)
  {
    float32x4_t in1 = vld1q_f32(&out_data[i]);
    float32x4_t ans = vmulq_n_f32(in1,(float32_t)sum);
    vst1q_f32(&out_data[i],ans);
  }
  for(i=limit;i<limit+leftover;i++){
    out_data[i]*=sum;
  }

  return;
}
