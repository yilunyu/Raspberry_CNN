#include "operation.h"
#include "tensor.h"
//#include "controller.h"
#include <assert.h>
#include <vector>
#include <new>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#define DEBUG

void string_2_intarr(std::string line,int* output){
  std::stringstream ss (line);
  int cnt = 0;
  while(ss.good()){
    std::string sub;
    getline(ss,sub,' ');
    output[cnt] = std::stof(sub);
    cnt++;
  }
}
void string_2_doublearr(std::string line,double* output){
  std::stringstream ss (line);
  int cnt = 0;
  while(ss.good()){
    std::string sub;
    getline(ss,sub,' ');
    output[cnt] = std::stof(sub);
    cnt++;
  }
}

void test_conv(){
  std::ifstream ins("in_conv.txt");
  std::ofstream outs("out_conv.txt");
  std::string line;
  double *data_1; //= new double[25];
  double *w_1;
  int *dim1 = new int[4];
  int *dim2 = new int[4];
  //Tensor in_1 = Tensor(5,5,1,1,data_1,name_1);
  int n_tests = 0;
  std::string name_1 ("input_1");
  std::string name_w1 ("w_1");
  if(ins.is_open() && outs.is_open()){
    getline(ins,line);
    n_tests = std::stoi(line);
    for(int i =0;i<n_tests;i++){
      getline(ins,line);
      string_2_intarr(line,dim1);
      getline(ins,line);
      int data_len = std::stoi(line);
      data_1 = new double[data_len];
      getline(ins,line);
      string_2_doublearr(line,data_1);

      getline(ins,line);
      string_2_intarr(line,dim2);
      getline(ins,line);
      data_len = std::stoi(line);
      w_1 = new double[data_len];
      getline(ins,line);
      string_2_doublearr(line,w_1);
      Tensor t_1 = Tensor(dim1[0],dim1[1],dim1[2],dim1[3],data_1,name_1);
      Tensor t_w1 = Tensor(dim2[0],dim2[1],dim2[2],dim2[3],w_1,name_w1);
      std::vector<Tensor> tens;
      tens.push_back(t_1);
      tens.push_back(t_w1);
      Convolution conv(tens,"conv_1","conv_1_out");
      conv.apply_function();
      Tensor conv_out = conv.get_output();
      double* conv_data = conv_out.get_data();


      int conv_data_len = conv_out.dim*conv_out.height*conv_out.width*conv_out.num_filter;
      for(int j=0;j<conv_data_len;j++){
          std::cout << conv_data[j] <<' ';
      }
      std::cout <<'\n';

      delete[] data_1;
      delete[] w_1;

    }

  }
  //assert(false);
  /*
  std::string name_1 ("n_1");
  double *data_1 = new double[25];
  Tensor in_1 = Tensor(5,5,1,1,data_1,name_1);
  std::string name_2 ("n_2");
  double *data_2 = new double[81];
  Tensor in_1 = Tensor(9,9,1,1,data_2,name_2);
  std::string name_3 ("n_3");
  double *data_3 = new double[162];
  Tensor in_1 = Tensor(9,9,2,1,data_3,name_3);
  std::string name_4 ("n_4");
  double *data_4 = new double[50];
  Tensor in_1 = Tensor(5,5,2,1,data_4,name_4);


  double *data_w1 = new double[9];
  std::string name_w1 ("n_w1");
  Tensor w_1 = Tensor(3,3,1,1,data_w1,name_w1);
  double *data_w2 = new double[9];
  std::string name_w2 ("n_w2");
  Tensor w_2 = Tensor(3,3,2,1,data_w2,name_w2);
  double *data_w3 = new double[9];
  std::string name_w3 ("n_w3");
  Tensor w_3 = Tensor(3,3,1,2,data_w3,name_w3);
  double *data_w4 = new double[9];
  std::string name_w4 ("n_w4");
  Tensor w_4 = Tensor(3,3,2,2,data_w4,name_w4);


  delete[] data_1;
  delete[] data_w1;
  delete[] data_2;
  delete[] data_w2;
  delete[] data_3;
  delete[] data_w3;
  delete[] data_4;
  delete[] data_w4;*/
}

void test_full(){
  std::ifstream ins("in_full.txt");
  std::ofstream outs("out_full.txt");
  std::string line;
  double *data_1; //= new double[25];
  double *w_1;
  double *b_1;
  int *dim_2 = new int[4];
  //Tensor in_1 = Tensor(5,5,1,1,data_1,name_1);
  int n_tests = 0;
  std::string name_1 ("input_1");
  std::string name_w1 ("w_1");
  std::string name_b1 ("b_1");

  if(ins.is_open() && outs.is_open()){
    getline(ins,line);
    n_tests = std::stoi(line);
    for(int i =0;i<n_tests;i++){
      getline(ins,line);
      int data_len = std::stoi(line);
      data_1 = new double[data_len];
      getline(ins,line);
      string_2_doublearr(line,data_1);
      Tensor t_1 = Tensor(data_len,1,1,1,data_1,name_1);

      getline(ins,line);
      string_2_intarr(line,dim_2);
      getline(ins,line);
      data_len = std::stoi(line);
      w_1 = new double[data_len];
      getline(ins,line);
      string_2_doublearr(line,w_1);
      Tensor t_w1 = Tensor(dim_2[0],dim_2[1],dim_2[2],dim_2[3],w_1,name_w1);
      b_1 = new double[dim_2[0]];
      getline(ins,line);
      string_2_doublearr(line,b_1);
      Tensor t_b1 = Tensor(dim_2[0],1,1,1,b_1,name_b1);
      
      std::vector<Tensor> tens;
      tens.push_back(t_1);
      tens.push_back(t_w1);
      tens.push_back(t_b1);
      FC fc(tens,"fc_1","fc_1_out");
      fc.apply_function();
      Tensor fc_out = fc.get_output();
      double* fc_data = fc_out.get_data();


      int fc_data_len = fc_out.dim*fc_out.height*fc_out.width*fc_out.num_filter;
      for(int j=0;j<fc_data_len;j++){
          std::cout << fc_data[j] <<' ';
      }
      std::cout <<'\n';

      delete[] data_1;
      delete[] w_1;
      delete[] b_1;
    }

  }

}

void test_pool(){

  std::ifstream ins("in_pool.txt");
  std::ofstream outs("out_pool.txt");
  std::string line;
  double *data_1; //= new double[25];
  int *dim_1 = new int[4];
  //Tensor in_1 = Tensor(5,5,1,1,data_1,name_1);
  int n_tests = 0;
  std::string name_1 ("input_1");

  if(ins.is_open() && outs.is_open()){
    getline(ins,line);
    n_tests = std::stoi(line);
    for(int i =0;i<n_tests;i++){
      getline(ins,line);
      string_2_intarr(line,dim_1);
      getline(ins,line);
      int data_len = std::stoi(line);
      data_1 = new double[data_len];
      getline(ins,line);
      string_2_doublearr(line,data_1);
      Tensor t_1 = Tensor(dim_1[0],dim_1[1],dim_1[2],dim_1[3],data_1,name_1);

      
      std::vector<Tensor> tens;
      tens.push_back(t_1);
      Pooling pooling(tens,"pool_1","pool_1_out");
      pooling.apply_function();
      Tensor pooling_out = pooling.get_output();
      double* pooling_data = pooling_out.get_data();


      int pooling_data_len = pooling_out.dim*pooling_out.height*pooling_out.width*pooling_out.num_filter;
      for(int j=0;j<pooling_data_len;j++){
          std::cout << pooling_data[j] <<' ';
      }
      std::cout <<'\n';

      delete[] data_1;

    }

  }
}
int main(){
  test_conv();
  //test_full();
  //test_pool();
  return 0;
}
