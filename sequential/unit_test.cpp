#include "operation.h"
#include "tensor.h"
#include "controller.h"
#include <assert.h>
#include <vector>
#include <new>
#define DEBUG

#ifdef DEBUG

#endif

void test_conv(){
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
  return 0;
}
