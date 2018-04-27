#include "operation.h"
#include "tensor.h"
#include "controller.h"
#include <assert.h>
#include <vector>
#include <new>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#define DEBUG

void string_2_doublearr(string line,double* output){
  stringstream ss (line);
  int cnt = 0;
  while(ss.good()){
    string sub;
    getline(ss,sub,' ');
    output[cnt] = atof(sub);
    cnt++;
  }   
}

void test_conv(){
  ifstream ins("in_conv.txt");
  ofstream outs("out_conv.txt");
  string line;
  std::string name_1 ("inp");
  double *data_1; //= new double[25];
  double *w_1;
  double *dim1 = new double[4];
  double *dim2 = new double[4];
  //Tensor in_1 = Tensor(5,5,1,1,data_1,name_1);
  int n_tests = 0;
  std::string name_1 ("input_1");
  std::string name_w1 ("w_1");
  if(ins.is_open() && outs.is_open()){
    getline(ins,line);
    n_tests = atoi(line);
    for(int i =0;i<n_tests;i++){
      getline(ins,line);
      string_2_doublearr(line,dim1);
      getline(ins,line);
      int data_len = atoi(line);
      data_1 = new double[data_len];
      getline(ins,line);
      string_2_doublearr(line,data_1);

      getline(ins,line);
      string_2_doublearr(line,dim2);
      getline(ins,line);
      data_len = atoi(line);
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
        cout << conv_data[j];
      }
      delete[] data_1;
      delete[] w_1;
    }
  }
  assert(false)
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
  delete[] data_w4;
}

int main(){
  test_conv();
  return 0;
}
