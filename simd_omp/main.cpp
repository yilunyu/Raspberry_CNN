#include "operation.h"
#include "tensor.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <new>
#include <fstream>
#include <string>
#include <time.h> 

void string_2_intarr(std::string line,int* output){
  std::stringstream ss (line);
  int cnt = 0;
  while(ss.good()){
    std::string sub;
    getline(ss,sub,' ');
    output[cnt] = std::stoi(sub);
    cnt++;
  }
}
void string_2_doublearr(std::string line,ne10_float32_t* output){
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
  ne10_float32_t *data_1; //= new double[25];
  ne10_float32_t *w_1;
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
      data_1 = new ne10_float32_t[data_len];
      getline(ins,line);
      string_2_doublearr(line,data_1);

      getline(ins,line);
      string_2_intarr(line,dim2);
      getline(ins,line);
      data_len = std::stoi(line);
      w_1 = new ne10_float32_t[data_len];
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
      ne10_float32_t* conv_data = conv_out.get_data();


      int conv_data_len = conv_out.dim*conv_out.height*conv_out.width*conv_out.num_filter;
      for(int j=0;j<conv_data_len;j++){
          //std::cout << conv_data[j] <<' ';
          outs << conv_data[j] <<' ';
      }
      //std::cout <<'\n';
      outs <<'\n';

      delete[] data_1;
      delete[] w_1;

    }
    ins.close();
    outs.close();
  }
}

void test_pool(){

  std::ifstream ins("in_pool.txt");
  std::ofstream outs("out_pool.txt");
  std::string line;
  ne10_float32_t *data_1; //= new double[25];
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
      data_1 = new ne10_float32_t[data_len];
      getline(ins,line);
      string_2_doublearr(line,data_1);
      Tensor t_1 = Tensor(dim_1[0],dim_1[1],dim_1[2],dim_1[3],data_1,name_1);

      
      std::vector<Tensor> tens;
      tens.push_back(t_1);
      Pooling pooling(tens,"pool_1","pool_1_out");
      pooling.apply_function();
      Tensor pooling_out = pooling.get_output();
      ne10_float32_t* pooling_data = pooling_out.get_data();

      
      int pooling_data_len = pooling_out.dim*pooling_out.height*pooling_out.width*pooling_out.num_filter;
      for(int j=0;j<pooling_data_len;j++){
          //std::cout << pooling_data[j] <<' ';
          outs << pooling_data[j] <<' ';
    
      }
      outs<<'\n';
      //std::cout <<'\n';

      delete[] data_1;

    }
    ins.close();
    outs.close();
  }
}

void test_full(){
  std::ifstream ins("in_fc.txt");
  std::ofstream outs("out_fc.txt");
  std::string line;
  ne10_float32_t *data_1; 
  ne10_float32_t *w_1;
  ne10_float32_t *b_1;
  ne10_int32_t *dim_2 = new ne10_int32_t[4];
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
      data_1 = new ne10_float32_t[data_len];
      getline(ins,line);
      string_2_doublearr(line,data_1);
      Tensor t_1 = Tensor(data_len,1,1,1,data_1,name_1);

      getline(ins,line);
      string_2_intarr(line,dim_2);
      getline(ins,line);
      data_len = std::stoi(line);
      w_1 = new ne10_float32_t[data_len];
      getline(ins,line);
      string_2_doublearr(line,w_1);
      Tensor t_w1 = Tensor(dim_2[0],dim_2[1],dim_2[2],dim_2[3],w_1,name_w1);
      b_1 = new ne10_float32_t[dim_2[0]];
      getline(ins,line);
      string_2_doublearr(line,b_1);
      Tensor t_b1 = Tensor(1,dim_2[0],1,1,b_1,name_b1);
      
      std::vector<Tensor> tens;
      tens.push_back(t_1);
      tens.push_back(t_w1);
      tens.push_back(t_b1);
      //std::cout<<w_1<<' '<<t_w1.get_data()<<'\n';
      
      //t_1.print_t();
      //t_w1.print_t();
      //t_b1.print_t();
      
      FC fc(tens,"fc_1","fc_1_out");
      fc.apply_function();
      Tensor fc_out = fc.get_output();
      ne10_float32_t* fc_data = fc_out.get_data();

      //fc_out.print_t();
      ne10_int32_t fc_data_len = fc_out.dim*fc_out.height*fc_out.width*fc_out.num_filter;
      for(int j=0;j<fc_data_len;j++){
        //  std::cout << fc_data[j] <<' ';
	outs << fc_data[j] <<' ';
      }
      //std::cout <<'\n';
      outs<<'\n';
      delete[] data_1;
      delete [] w_1;
      delete[] b_1;
    }
    ins.close();
    outs.close();

  }

}

void test_softmax(){

  std::ifstream ins("in_soft.txt");
  std::ofstream outs("out_soft.txt");
  std::string line;
  ne10_float32_t *data_1; //= new ne10_float32_t[25];
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
      data_1 = new ne10_float32_t[data_len];
      getline(ins,line);
      string_2_doublearr(line,data_1);
      Tensor t_1 = Tensor(dim_1[0],dim_1[1],dim_1[2],dim_1[3],data_1,name_1);

      Softmax soft(t_1,"soft_1","soft_1_out");
      //t_1.print_t();
      soft.apply_function();
      //t_1.print_t();
      Tensor soft_out = soft.get_output();
      ne10_float32_t* soft_data = soft_out.get_data();

      
      int soft_data_len = soft_out.dim*soft_out.height*soft_out.width*soft_out.num_filter;
      for(int j=0;j<soft_data_len;j++){
          //std::cout << pooling_data[j] <<' ';
          outs << soft_data[j] <<' ';
    
      }
      outs<<'\n';
      //std::cout <<'\n';

      delete[] data_1;

    }
    ins.close();
    outs.close();
  }
}

int main(){
  time_t time0, time1;
  time(&time0);
  test_conv();
  time(&time1);
  double seconds = time1 -time0;
  printf("Conv Time: %f\n", seconds);

  time(&time0);
  test_full();
  time(&time1);
  seconds = time1 -time0;
  printf("Full Time: %f\n", seconds);

  time(&time0);
  test_softmax();
  time(&time1);
  seconds = time1 -time0;
  printf("Soft Time: %f\n", seconds);

  time(&time0);
  test_pool();
  time(&time1);
  seconds = time1 -time0;
  printf("Pool Time: %f\n", seconds);

  return 0;
}
