#include "operation.h"
#include "tensor.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include "controller.h"
#include <unordered_map>
#include <assert.h>
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
void string_2_doublearr(std::string line,double* output){
  std::stringstream ss (line);
  int cnt = 0;
  while(ss.good()){
    std::string sub;
    getline(ss,sub,' ');
    if(sub.compare("")==0){
      continue;
    }
    output[cnt] = std::stof(sub);
    cnt++;
  }
}

Controller* load_network(std::string filename){
  std::ifstream ins(filename);
  std::string line;
  Controller *controller = new Controller();
  int dim[4]; 
  if(ins.is_open()){
    while(getline(ins,line)){
      if(line.compare("Weight")==0){
        std::string id;
    	getline(ins,id);
	for(int i=0;i<4;i++)
	{
	  dim[i] = 1.;
	}
	getline(ins,line);
	string_2_intarr(line,dim);
	double *data = new double[dim[0]*dim[1]*dim[2]*dim[3]];
	getline(ins,line);
	string_2_doublearr(line,data);
	Tensor t = Tensor(dim[0],dim[1],dim[2],dim[3],data,id);
	controller->add_tensor(t);
      }
      else if(line.compare("Input")==0){
        std::string id;
    	getline(ins,id);
	for(int i=0;i<4;i++)
	{
	  dim[i] = 1.;
	}
	getline(ins,line);
	string_2_intarr(line,dim);
	Tensor t = Tensor(dim[0],dim[1],dim[2],dim[3],id);
      	controller->add_tensor(t);
      }
      else{
	getline(ins,line);
  	std::stringstream ss (line);
	std::string op_type;
	getline(ss,op_type,' ');
	std::string op_output;
	getline(ss,op_output,' ');
	if(op_type.compare("Softmax")==0 || op_type.compare("Relu")==0){
	  std::string sub;
	  getline(ss,sub,' ');
	  Tensor curr = controller->get_tensor(sub);
	  if(op_type.compare("Softmax")==0){
	    Softmax *softmax = new Softmax(curr,"Soft_1",op_output);
	    controller->add_op(softmax);
	  }
	  else{
	    Relu* relu = new Relu(curr,"Relu_1",op_output);
	    controller->add_op(relu);
	  }
	  continue;
	}
	std::vector<Tensor> inputs;
	while(ss.good()){
    	  std::string sub;
    	  getline(ss,sub,' ');
  	  Tensor curr = controller->get_tensor(sub);
	  inputs.push_back(curr);
	}
	if(op_type.compare("FC")==0){
	  FC *fc = new FC(inputs,"FC_1",op_output);
	  controller->add_op(fc);
	}
	else if(op_type.compare("Convolution")==0){
	  Convolution *convolution = new Convolution(inputs,"Conv_1",op_output);
	  controller->add_op(convolution);
	}
	else if(op_type.compare("Pooling")==0){
	  Pooling *pooling = new Pooling(inputs,"Pool_1",op_output);
	  controller->add_op(pooling);
	}
	else if(op_type.compare("Flatten")==0){
	  Flatten *flatten = new Flatten(inputs,"Flatten_1",op_output);
	  controller->add_op(flatten);
	}
	else{
	  std::cout<<"Operation does not exist\n";
	  assert(false);
	}
      }
    }
  }
  return controller;
}

int main(){
  std::string filename("./tensorflow_model/test_file");
  Controller* controller = load_network(filename);
  
  Tensor t = controller->get_tensor(std::string("x"));
  double* t_data = t.get_data();
  for(int i=0;i<t.width*t.height*t.dim*t.num_filter;i++)
  {
    t_data[i] = i%6-2;
  }
  
  //Tensor t2 = controller->get_tensor(std::string("W"));
  //t2.print_t();
  //std::cout<<t2.height<<' '<<t2.width<<' '<<t2.num_filter*t2.dim<<'\n';
  //Tensor t3 = controller->get_tensor(std::string("b"));
  //t3.print_t();
  //std::cout<<t3.height<<' '<<t3.width<<' '<<t3.num_filter*t3.dim<<'\n';
  
  controller->forward_pass();
  Tensor output = controller->get_tensor(std::string("FC_out"));
  output.print_t(); 
  return 0;
}
