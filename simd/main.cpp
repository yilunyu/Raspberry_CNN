#include "operation.h"
#include "tensor.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <new>
#include <fstream>
#include <string>
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
int main(){
  test_full();
  return 0;
}
